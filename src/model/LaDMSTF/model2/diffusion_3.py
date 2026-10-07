import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from diffusers import EDMEulerScheduler, EDMDPMSolverMultistepScheduler, DDIMScheduler
import tqdm
import math

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class EDMDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        num_train_timesteps=1000,
        loss_type='l1',
        prediction_type='sample',
        sampling_timesteps = 100,
    ):
        super().__init__()
        self.T1 = 100
        self.T2 = 500
        self.sample_T1 = (self.T1 * sampling_timesteps // num_train_timesteps) if self.T1 > 200 else 15
        self.sample_T2 = self.T2 * sampling_timesteps // num_train_timesteps
        print(f"sample_timesteps: {sampling_timesteps}")
        print(f"sample_T1: {self.sample_T1}, sample_T2: {self.sample_T2}")

        self.model = model
        self.image_size = image_size
        self.loss_type = loss_type
        self.prediction_type = prediction_type
        self.num_train_timesteps = num_train_timesteps
        self.sampling_timesteps = sampling_timesteps
        self.init_para(timesteps=num_train_timesteps)


    def laplacian_pyramid(self, img, levels=3):
        pyramid = []
        for i in range(levels):
            down = F.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=False)
            up = F.interpolate(down, size=img.shape[2:], mode='bilinear', align_corners=False)
            pyramid.append(img - up)
            img = down
        pyramid.append(img)
        return pyramid[0], pyramid[1], pyramid[2]

    def init_para(self, beta_schedule='cosine', timesteps=1000):
        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        if beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        register_buffer(
            'alpha_3', torch.sqrt(1.0 / alphas_cumprod - 1)
        )
        register_buffer(
            'alpha_1', torch.sqrt(1.0 / alphas_cumprod - 1) * self.T1 / 1000
        )
        register_buffer(
            'alpha_2', torch.sqrt(1.0 / alphas_cumprod - 1) * self.T2 / 1000
        )
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)


        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer(
            'sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            'sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1)
        )


        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        register_buffer('posterior_variance', posterior_variance)

        register_buffer(
            'posterior_log_variance_clipped',
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            'posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            'posterior_mean_coef2',
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def get_x0_t(self, x1, x2, x3 , t):
        print("0 的时候是什么呢？")
        alpha_1 = extract(self.alpha_1, t, x1.shape)
        alpha_2 = extract(self.alpha_2, t, x2.shape)
        alpha_3 = extract(self.alpha_3, t, x3.shape)

        x0_t = x3 * alpha_3
        if 'x2' in self.mode:
            x0_t = F.interpolate(x0_t, scale_factor=2, mode='bilinear', align_corners=False)
            x0_t = x0_t + x2 * alpha_2
        if 'x1' in self.mode:
            x0_t = F.interpolate(x0_t, scale_factor=2, mode='bilinear', align_corners=False)
            x0_t = x0_t + x1 * alpha_1
        return x0_t

    def get_xt(self, x0_t, t, model=3, scale=1, noise = None):
        if model == 2:
            x_t = x0_t + extract(self.sqrt_one_minus_alphas_cumprod, t, x0_t.shape) * noise * scale
        elif model == 1:
            x_t = x0_t + extract(self.sqrt_one_minus_alphas_cumprod, t, x0_t.shape) * noise * scale
        else:
            x_t = x0_t + extract(self.sqrt_one_minus_alphas_cumprod, t, x0_t.shape) * torch.randn_like(x0_t) * scale
        return x_t

    def get_condition(self, coarse_img_01, coarse_img_02, fine_img_01):
        condition = torch.cat([coarse_img_01, coarse_img_02, fine_img_01], dim=1)
        if self.mode == 'x1+x2+x3':
            condition =  condition
        if self.mode == 'x2+x3':
            condition = F.interpolate(condition, scale_factor=0.5, mode='bilinear', align_corners=False)
        if self.mode == 'x3':
            condition = F.interpolate(condition, scale_factor=0.25, mode='bilinear', align_corners=False)
        return condition


    def p_losses(self, coarse_img_01, coarse_img_02, fine_img_01, fine_img_02, t):
        x1, x2, x3 = self.laplacian_pyramid(fine_img_02, levels=3)

        x0_t = self.get_x0_t(x1, x2, x3, t)

        x_t = self.get_xt(x0_t, t)
        condition = self.get_condition(coarse_img_01, coarse_img_02, fine_img_01)

        outputs = self.model(x_t, t, condition)

        loss = self.loss_fn(outputs, x0_t)

        return loss

    def forward(self, coarse_img_01, coarse_img_02, fine_img_01, fine_img_02):
        (
            b,
            c,
            h,
            w,
            device,
            img_size,
        ) = (
            *coarse_img_01.shape,
            coarse_img_01.device,
            self.image_size,
        )
        assert (
            h == img_size and w == img_size
        ), f'height and width of image must be {img_size}'

        self.mode = 'x3'
        t = torch.randint(0, self.num_train_timesteps, (b,), device=device).long()

        return self.p_losses(coarse_img_01, coarse_img_02, fine_img_01, fine_img_02, t)

    @torch.no_grad()
    def p_sample_loop(
        self, coarse_img_01, coarse_img_02, fine_img_01, noisy_pyramid
    ):
        noise_1, noise_2, noise_3 = noisy_pyramid[0], noisy_pyramid[1], noisy_pyramid[2]
        noisy_image = torch.randn_like(fine_img_01)
        xt = noisy_image
        bs = fine_img_01.shape[0]
        self.mode = 'x3'
        for t in tqdm.tqdm(
            reversed(range(0, self.sampling_timesteps)),
            desc='sampling loop time step',
            total=self.sampling_timesteps,
        ):

            batched_times = torch.full(
                    (bs,),
                    t,
                    device=fine_img_01.device,
                    dtype=torch.long,
            )
            condition = self.get_condition(coarse_img_01, coarse_img_02, fine_img_01)
            x0_t = self.model(
                    xt,
                    batched_times,
                    condition,
            )

            if t == 0:
                xt = x0_t
            else:
                xt = self.get_xt(x0_t, batched_times)

        return xt

    @torch.no_grad()
    def sample(self, coarse_img_01, coarse_img_02, fine_img_01):

        noisy_img = torch.randn_like(fine_img_01)

        noise_1, noise_2, noise_3 = self.laplacian_pyramid(noisy_img, levels=3)
        noise_pyramid = [noise_1, noise_2, noise_3]

        return self.p_sample_loop(
            coarse_img_01,
            coarse_img_02,
            fine_img_01,
            noise_pyramid,
        )
