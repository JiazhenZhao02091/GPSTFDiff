import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import cv2
import numpy as np
from functools import partial
from collections import namedtuple
from einops import reduce
from tqdm.auto import tqdm


ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def exists(x):
    return x is not None

def identity(t, *args, **kwargs):
    return t

def cv2_interpolate(tensor, size=None, scale_factor=None, mode="bilinear", align_corners=None):
    del align_corners
    h, w = tensor.shape[-2:]
    if size is None:
        if isinstance(scale_factor, (tuple, list)):
            scale_h, scale_w = scale_factor
        else:
            scale_h = scale_w = scale_factor
        out_h, out_w = int(h * float(scale_h)), int(w * float(scale_w))
    else:
        out_h, out_w = size
        out_h, out_w = int(out_h), int(out_w)
    if (out_h, out_w) == (h, w):
        return tensor
    interpolation = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
    }[mode]
    device, dtype = tensor.device, tensor.dtype
    array = tensor.detach().contiguous().cpu().float().numpy()
    b, c, in_h, in_w = array.shape
    array = array.reshape(b * c, in_h, in_w)
    resized = np.stack([cv2.resize(img, (out_w, out_h), interpolation=interpolation) for img in array], axis=0)
    resized = resized.reshape(b, c, out_h, out_w)
    return torch.from_numpy(resized).to(device=device, dtype=dtype)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Diffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        num_train_timesteps=1000,         # 训练时从 0 - 999 中随机采样时间步t，噪声添加的精细粒度
        sampling_timesteps = 100,
        loss_type='l1',
        objective='pred_x0',
        mode = "x1+x2+x3",
        T1 = 100,
        T2 = 150,
        ddim_sampling_eta=1.0,
        beta_schedule = 'cosine',
        p2_loss_weight_gamma=0.0,  # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k=1,
        
        model_x3 = None,
        model_x2_x3 = None,
        model_x1_x2_x3 = None,
    ):
        super().__init__()
        # init model
        self.mode = mode
        self.model_x3 = model_x3
        self.model_x2_x3 = model_x2_x3
        self.model_x1_x2_x3 = model_x1_x2_x3

        self.T1 = T1
        self.T2 = T2
        self.T3 = num_train_timesteps
        self.sample_T1 = T1 / num_train_timesteps * sampling_timesteps
        self.sample_T2 = T2 / num_train_timesteps * sampling_timesteps

        self.ddim_sampling_eta = ddim_sampling_eta
        self.model = model
        self.image_size = image_size
        self.loss_type = loss_type
        self.num_train_timesteps = num_train_timesteps
        self.sampling_timesteps = sampling_timesteps
        timesteps = num_train_timesteps
        self.timesteps = timesteps
        self.num_timesteps = timesteps
        self.p2_loss_weight_gamma = p2_loss_weight_gamma
        self.p2_loss_weight_k = p2_loss_weight_k
        self.self_condition = False
        self.objective = objective
        self.init_para(timesteps=num_train_timesteps)
        
    " 参数 "
    def generate_decay_tensors_cosine(
            self,
            t1: int, 
            t2: int, 
            total_steps: int = 1000, 
            initial_value: float = 1.0
        ):
        if not (1 <= t1 <= total_steps) or not (1 <= t2 <= total_steps):
            raise ValueError(f"t1 和 t2 必须在 [1, {total_steps}] 范围内。")
        time_steps = torch.arange(total_steps, dtype=torch.float32)
        decay_duration_1 = float(total_steps)
        arg_1 = (time_steps / decay_duration_1) * torch.pi
        list1 = (initial_value / 2.0) * (1 + torch.cos(arg_1))
        decay_duration_2 = float(t1)
        arg_2 = (time_steps / decay_duration_2) * torch.pi
        list2 = (initial_value / 2.0) * (1 + torch.cos(arg_2))
        mask_2 = time_steps >= t1
        list2[mask_2] = 0.0
        decay_duration_3 = float(t2)
        arg_3 = (time_steps / decay_duration_3) * torch.pi
        list3 = (initial_value / 2.0) * (1 + torch.cos(arg_3))
        mask_3 = time_steps >= t2
        list3[mask_3] = 0.0
        
        return list1, list2, list3
    
    def init_para(self, beta_schedule='cosine', timesteps=1000):
        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        if beta_schedule == 'cosine':
            betas = cosine_beta_schedule(self.num_train_timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        
        alphas = 1.0 - betas                            # alpha = 1 - beta
        alphas_cumprod = torch.cumprod(alphas, dim=0)   # alpha_cumprod = alpha_0 * alpha_1 * ... * alpha_t
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        alpha_3, alpha_1, alpha_2 = self.generate_decay_tensors_cosine(t1 = self.T1, t2 = self.T2, total_steps = timesteps)

        register_buffer('alpha_1', alpha_1)
        register_buffer('alpha_2', alpha_2)
        register_buffer('alpha_3', alpha_3)
        print(f"alpha_3[400]: {alpha_3[400]}, alpha_1[400]: {alpha_1[400]}, alpha_2[400]: {alpha_2[400]}")
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer(
            'sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            'sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

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

        # calculate p2 reweighting
        
        register_buffer(
            'p2_loss_weight',
            (self.p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -self.p2_loss_weight_gamma,
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    # 同时返回预测的两个值：噪声和去噪后的图像
    def model_predictions(
        self,
        coarse_img_01,
        coarse_img_02,
        fine_img_01,
        noisy_fine_img_02,
        t,
        x_self_cond=None,
        clip_x_start=False,
    ):
        condition = self.get_condition(coarse_img_01, coarse_img_02, fine_img_01)
        model_output = self.model(
            noisy_fine_img_02, t, condition
        )
        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        )

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(noisy_fine_img_02, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(noisy_fine_img_02, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(
        self,
        coarse_img_01,
        coarse_img_02,
        fine_img_01,
        noisy_fine_img_02,
        t,
        x_self_cond=None,
        clip_denoised=True,
    ):
        preds = self.model_predictions(
            coarse_img_01, coarse_img_02, fine_img_01, noisy_fine_img_02, t, x_self_cond
        )
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=noisy_fine_img_02, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def get_condition(self, coarse_img_01, coarse_img_02, fine_img_01):
        condition = torch.cat([coarse_img_01, coarse_img_02, fine_img_01], dim=1)
        if self.mode == 'x1+x2+x3':
            condition =  condition
        if self.mode == 'x2+x3':
            condition = cv2_interpolate(condition, scale_factor=0.5, mode='bilinear', align_corners=False)
        if self.mode == 'x3':
            condition = cv2_interpolate(condition, scale_factor=0.25, mode='bilinear', align_corners=False)
        return condition

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def get_x0_t(self, x1, x2, x3 , t):
        " alpha 是个list "
        alpha_1 = extract(self.alpha_1, t, x1.shape)
        alpha_2 = extract(self.alpha_2, t, x2.shape)
        alpha_3 = extract(self.alpha_3, t, x3.shape)

        if self.mode == "x3":
            x0_t = x3 * alpha_3
        elif self.mode == "x2+x3":
            x3 = cv2_interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
            x0_t = x3 * alpha_3 + x2 * alpha_2
        else:   # x1 + x2 + x3
            x3 = cv2_interpolate(x3, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = cv2_interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x0_t = x3 * alpha_3 + x2 * alpha_2 + x1 * alpha_1

        return x0_t

    " 返回金字塔分解结果 "
    def laplacian_pyramid(self, img, levels=3):
        pyramid = []
        for i in range(levels):
            """
                Modify
            """
            if i == levels - 1:
                pyramid.append(img)
                break
            down = cv2_interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=False)
            up = cv2_interpolate(down, size=img.shape[2:], mode='bilinear', align_corners=False)
            pyramid.append(img - up)
            img = down
        pyramid.append(img)
        # print(f"pyramid 的长度 :{len(pyramid)}")
        return pyramid[0], pyramid[1], pyramid[2]

    def get_target(self, fine_img_02):
        x1, x2, x3 = self.laplacian_pyramid(fine_img_02, levels=3)
        if self.mode == "x1+x2+x3":
            return fine_img_02
        elif self.mode == "x2+x3":
            return x2 \
                 + cv2_interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        else: # x3
            return x3

    def p_losses(self, coarse_img_01, coarse_img_02, fine_img_01, fine_img_02, t, noise=None):
        b, c, h, w = fine_img_02.shape
        # noise = default(noise, lambda: torch.randn_like(fine_img_02))
        # noisy_fine_img_02 = self.q_sample(x_start=fine_img_02, t=t, noise=noise)
        x1, x2, x3 = self.laplacian_pyramid(fine_img_02, levels=3)
        x0_t = self.get_x0_t(x1, x2, x3, t)
        noise = default(noise, lambda: torch.randn_like(x0_t))
        
        noisy_fine_img_02 = self.q_sample(x_start=x0_t, t=t, noise=noise)
        condition = self.get_condition(coarse_img_01, coarse_img_02, fine_img_01)

        model_out = self.model(
            noisy_fine_img_02, t, condition
        )
        # target = self.get_target(fine_img_02)  # default is x0
        target = x0_t # TODO:x0_t or fine_img_02?
        loss = self.loss_fn(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

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

        if self.mode == "x1+x2+x3":
            t = torch.randint(0, self.T1, (b,), device=device).long()
        elif self.mode == "x2+x3":
            t = torch.randint(0, self.T2, (b,), device=device).long()
        else:
            t = torch.randint(0, self.T3, (b,), device=device).long()
        
        " 返回损失值 "
        return self.p_losses(coarse_img_01, coarse_img_02, fine_img_01, fine_img_02, t)

    @torch.no_grad()
    def p_sample(
        self,
        coarse_img_01,
        coarse_img_02,
        fine_img_01,
        noisy_fine_img_02,
        t: int,
        x_self_cond=None,
        clip_denoised=True,
    ):
        b, *_, device = *noisy_fine_img_02.shape, noisy_fine_img_02.device
        batched_times = torch.full(
            (noisy_fine_img_02.shape[0],),
            t,
            device=noisy_fine_img_02.device,
            dtype=torch.long,
        )
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            coarse_img_01,
            coarse_img_02,
            fine_img_01,
            noisy_fine_img_02,
            t=batched_times,
            x_self_cond=x_self_cond,
            clip_denoised=clip_denoised,
        )
        noise = (
            torch.randn_like(noisy_fine_img_02) if t > 0 else 0.0
        )  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(
        self, coarse_img_01, coarse_img_02, fine_img_01, noisy_fine_img_02
    ):
        # noisy_fine_img_02 = torch.randn_like(fine_img_01)

        x_start = None

        for t in tqdm(
            reversed(range(0, self.num_train_timesteps)),
            desc='sampling loop time step',
            total=self.num_train_timesteps,
        ):
            self_cond = x_start if self.self_condition else None
            noisy_fine_img_02, x_start = self.p_sample(
                coarse_img_01,
                coarse_img_02,
                fine_img_01,
                noisy_fine_img_02,
                t,
                self_cond,
            )

        return noisy_fine_img_02

    @torch.no_grad()
    def ddim_sample(
        self,
        coarse_img_01,  
        coarse_img_02,
        fine_img_01,
        noisy_fine_img_02,
        clip_denoised=True,
    ):
        # TODO: 三阶段采样，只用修改这里的sample函数，不修改别的。
        print(f"DDIM sampleing ...")
        batch, device, total_timesteps, sampling_timesteps, eta, objective = (
            noisy_fine_img_02.shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
            self.objective,
        )

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x_start = None
        i = sampling_timesteps # 100

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(
                coarse_img_01,
                coarse_img_02,
                fine_img_01,
                noisy_fine_img_02,
                time_cond,
                self_cond,
                clip_x_start=clip_denoised,
            )

            if time_next < 0:
                noisy_fine_img_02 = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()

            # switch state mode.
            i = i - 1
            if i == self.sample_T2:
                print(f"Sampling stage change: T3 -> T2 at step {i}")
                print(f"Current mode: {self.mode}")
                self.mode = "x2+x3"
                self.model = self.model_x2_x3

                scale = 0.5
                x_start_up = cv2_interpolate(x_start, scale_factor=2, mode='bilinear', align_corners=False)
                noise = torch.randn_like(x_start_up)
                # noisy_fine_img_02 = noisy_fine_img_02 + sigma * scale
                noisy_fine_img_02 = x_start_up + noise * (1 - alpha_next).sqrt() * scale
                continue
            
            if i == self.sample_T1:
                print(f"Sampling stage change: T2 -> T1 at step {i}")
                print(f"Current mode: {self.mode}")
                self.mode = "x1+x2+x3"
                self.model = self.model_x1_x2_x3

                scale = 0.5
                x_start_up = cv2_interpolate(x_start, scale_factor=2, mode='bilinear', align_corners=False)
                noise = torch.randn_like(x_start_up)
                noisy_fine_img_02 = x_start_up + noise * (1 - alpha_next).sqrt() * scale
                continue

            noise = torch.randn_like(noisy_fine_img_02)
            noisy_fine_img_02 = (
                x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
            )          

        return noisy_fine_img_02
    
    @torch.no_grad()
    def sample(self, coarse_img_01, coarse_img_02, fine_img_01):
        b,c,h,w = fine_img_01.shape
        noisy_fine_img_02 = torch.randn(b,c,h//4, w//4).to(fine_img_01.device)
        self.mode = "x3"        #    从初始状态开始-协同condition
        self.model = self.model_x3
        return self.ddim_sample(
            coarse_img_01,
            coarse_img_02,
            fine_img_01,
            noisy_fine_img_02,
        )