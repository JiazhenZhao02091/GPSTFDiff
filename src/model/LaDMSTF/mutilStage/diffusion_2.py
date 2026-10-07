import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from diffusers import EDMEulerScheduler, EDMDPMSolverMultistepScheduler, DDIMScheduler
import tqdm

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
        sampling_timesteps=None,
        loss_type='l1',
        prediction_type='sample',

        scheduer_type = "DDIM"
    ):
        super().__init__()

        self.model = model
        self.image_size = image_size
        self.loss_type = loss_type
        self.prediction_type = prediction_type
        self.sampling_timesteps = sampling_timesteps or 100
        if scheduler_type == "DDIM":
            self.scheduler = DDIMScheduler(
                num_train_timesteps=num_train_timesteps,
                prediction_type = prediction_type
            )
            print("使用的去噪网络为 DDIM Scheduler.")
        else:
            self.scheduler = EDMEulerScheduler(
                num_train_timesteps=num_train_timesteps,
                prediction_type = prediction_type
            )

    def laplacian_pyramid(self, img, levels=3):
        pyramid = []
        for i in range(levels):
            down = F.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=False)
            up = F.interpolate(down, size=img.shape[2:], mode='bilinear', align_corners=False)
            pyramid.append(img - up)
            img = down
        pyramid.append(img)
        return pyramid


        self.init_alpha_time_reduce()

    def init_para(self):
        betas = cosine_beta_schedule(1000)
        register_buffer(
            'alpha_1', torch.sqrt(1.0 / alphas_cumprod - 1)
        )
        register_buffer(
            'alpha_2', torch.sqrt(1.0 / alphas_cumprod - 1)
        )
        register_buffer(
            'alpha_3', torch.sqrt(1.0 / alphas_cumprod - 1)
        )


    def _down_factor_2(self, x):
            return self.downsample(self.downsample(x))
    def _down_factor_1(self, x):
        return self.downsample(x)

    def init_alpha_time_reduce(self):
        alpha_1 = []
        alpha_2 = []
        alpha_3 = []
        for i in range(self.scheduler.num_train_timesteps):
            alpha_1.append(1 - i / 250 if i < 250 else 0)
            alpha_2.append(1 - i / 500 if i < 500 else 0)
            alpha_3.append(1 - i / 1000 if i < 1000 else 0)

        self.alpha_1 = torch.tensor(alpha_1, dtype=torch.float32)
        self.alpha_2 = torch.tensor(alpha_2, dtype=torch.float32)
        self.alpha_3 = torch.tensor(alpha_3, dtype=torch.float32)
        self.t2 = 500
        self.t1 = 250
        print("alpha init successful!!")
        pass


    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')


    def _add(self, x1, x2, x3, t, c1, c2, f1, f2):
        if isinstance(t, torch.Tensor):
            if t.ndim > 0:
                t = t[0].item()
            else:
                t = int(t.item())
        model_input_list = []

        if t > self.t2:
            model_input_list.append(self._down_factor_2(c1))
            model_input_list.append(self._down_factor_2(c2))
            model_input_list.append(self._down_factor_2(f1))
            model_input_list.append(self._down_factor_2(f2))

            return float(self.alpha_3[t]) * x3, model_input_list
        elif t > self.t1 and t <= self.t2:
            model_input_list.append(self._down_factor_1(c1))
            model_input_list.append(self._down_factor_1(c2))
            model_input_list.append(self._down_factor_1(f1))
            model_input_list.append(self._down_factor_1(f2))

            x3_upsample = self.upsample(float(self.alpha_3[t]) * x3)
            return x3_upsample + float(self.alpha_2[t]) * x2, model_input_list
        else:
            model_input_list.append(c1)
            model_input_list.append(c2)
            model_input_list.append(f1)
            model_input_list.append(f2)

            x3_upsample = self.upsample(float(self.alpha_3[t]) * x3)
            x2_upsample = self.upsample(x3_upsample + float(self.alpha_2[t]) * x2)
            return float(self.alpha_2[t]) * x1 + x2_upsample, model_input_list

    def _sub(self, x):
        x3 = self.downsample(self.downsample(x))
        x2 = self.downsample(x) - self.upsample(x3)
        x1 = x - self.upsample(x2)

        return x1, x2, x3

    def get_noise_image(self, x, t, c1, c2, f1, f2):


        x1, x2, x3 = self._sub(x)
        x_sum, model_input_list = self._add(x1, x2, x3, t, c1, c2, f1, f2)
        noise = torch.randn_like(x_sum)

        t_index_cpu = t.to("cpu")
        t = self.scheduler.timesteps[t_index_cpu]


        noise_image = self.scheduler.add_noise(x_sum, noise, t)
        return noise_image, noise, model_input_list

    def get_x0_t(self, x1, x2, x3 , t):
        x1, x2, x3 = self._sub(x)
        x_sum, model_input_list = self._add(x1, x2, x3, t, *model_input_list)
        return x_sum

    def p_losses(self, coarse_img_01, coarse_img_02, fine_img_01, fine_img_02, t):


        x1, x2, x3 = self.laplacian_pyramid(fine_img_02, levels=3)

        x0_t = self.get_x0_t(x1, x2, x3, t)

        batch_size = coarse_img_01.shape[0]
        timestep = torch.randint(self.t1, self.scheduler.num_train_timesteps, (batch_size,), device=fine_img_02.device)


        model_input = [coarse_img_01, coarse_img_02, fine_img_01, fine_img_02]
        noise_image, noise, model_input_list = self.get_noise_image(fine_img_02, timestep, *model_input)
        fine_img_02 = model_input_list[-1]
        model_input_list.remove(fine_img_02)
        outputs = self.model(*model_input_list, noise_image, timestep)


        if self.prediction_type == 'sample':
            loss = self.loss_fn(outputs, fine_img_02)
        else:
            loss = self.loss_fn(outputs, noise)


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
        t = torch.randint(0, self.scheduler.num_train_timesteps, (b,), device=device).long()


        return self.p_losses(coarse_img_01, coarse_img_02, fine_img_01, fine_img_02, t)


    @torch.no_grad()
    def sample(self, coarse_img_01, coarse_img_02, fine_img_01):
        coarse_img_01 = self._down_factor_1(coarse_img_01)
        coarse_img_02 = self._down_factor_1(coarse_img_02)
        fine_img_01   = self._down_factor_1(fine_img_01)

        noisy_image = torch.randn_like(fine_img_01)
        self.scheduler.set_timesteps(50)
        batch_size = fine_img_01.shape[0]

        for i,t in enumerate(tqdm.tqdm(self.scheduler.timesteps)):
            t_batch = torch.full((batch_size,), t, device=fine_img_01.device, dtype=torch.long)


            model_output = self.model(coarse_img_01, coarse_img_02, fine_img_01, noisy_image, t_batch)
            noisy_image = self.scheduler.step(model_output, t, noisy_image).prev_sample


        return noisy_image
