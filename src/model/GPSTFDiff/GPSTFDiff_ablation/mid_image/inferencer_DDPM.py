import math

from random import random
from functools import partial
from collections import namedtuple

from torch import nn
import torch

import torch.nn.functional as F

from einops import rearrange, reduce
from tqdm.auto import tqdm
import numpy as np
import tifffile
from pathlib import Path

"""
    * P2 Loss
    * self-cond 自条件
    * cosine 余弦调度
"""


# constants

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


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


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        sampling_timesteps=None,
        loss_type='l1',
        objective='pred_noise',
        beta_schedule='cosine',
        p2_loss_weight_gamma=0.0,
        p2_loss_weight_k=1,
        ddim_sampling_eta=1.0,
        save_interval=None,
        save_dir=None,
    ):
        super().__init__()
        # assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.learned_sinusoidal_cond

        self.save_dir = save_dir
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.objective = objective

        assert objective in {
            'pred_noise',
            'pred_x0',
        }, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start)'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

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
            (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -p2_loss_weight_gamma,
        )

        self.save_interval = save_interval

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

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
        model_output = self.model(
            coarse_img_01, coarse_img_02, fine_img_01, noisy_fine_img_02, t, x_self_cond
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
        self, 
        coarse_img_01, 
        coarse_img_02, 
        fine_img_01, 
        noisy_fine_img_02,
        save_dir=None,
        save_interval=None,
        normalize_scale=10000,
        normalize_mode=2,
        save_name_prefix='ddpm_mid'
    ):
        x_start = None
        for step, t in enumerate(tqdm(
            reversed(range(0, self.num_timesteps)),
            desc='sampling loop time step',
            total=self.num_timesteps,
        )):
            self_cond = x_start if self.self_condition else None
            noisy_fine_img_02, x_start = self.p_sample(
                coarse_img_01,
                coarse_img_02,
                fine_img_01,
                noisy_fine_img_02,
                t,
                self_cond,
            )

            # --- 保存 TIF 中间结果逻辑 ---
            if save_interval is not None and save_dir is not None:
                if (step + 1) % save_interval == 0  or step == 0:
                    save_tensor = x_start
                    save_img = save_tensor[0].cpu().numpy().transpose(1, 2, 0)
                    if normalize_mode == 1:
                        save_img = save_img * normalize_scale
                    elif normalize_mode == 2:
                        save_img = (save_img + 1.0) / 2.0 * normalize_scale
                    save_img = np.clip(save_img, 0, normalize_scale)
                    if normalize_scale == 255:
                        save_img = save_img.astype(np.uint8)
                    elif normalize_scale == 1:
                        save_img = save_img.astype(np.float32)
                    elif normalize_scale == 10000:
                        save_img = save_img.astype(np.uint16)
                    save_dir_path = Path(save_dir)
                    save_dir_path.mkdir(parents=True, exist_ok=True)
                    save_name = f"{save_name_prefix}_step{step+1}.tif"
                    save_img_path = save_dir_path / save_name
                    tifffile.imwrite(save_img_path, save_img)
            # -------------------------------

        return noisy_fine_img_02

    @torch.no_grad()
    def ddim_sample(
        self,
        coarse_img_01,
        coarse_img_02,
        fine_img_01,
        noisy_fine_img_02,
        clip_denoised=True,
        save_dir=None,
        normalize_scale=10000,
        normalize_mode=2,
        save_name_prefix='ddim_mid',
    ):
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

        # 使用 enumerate 获取当前 step
        for step, (time, time_next) in enumerate(tqdm(time_pairs, desc='sampling loop time step')):
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

            noise = torch.randn_like(noisy_fine_img_02)

            noisy_fine_img_02 = (
                x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
            )

            # --- 保存 TIF 中间结果逻辑 ---
            if self.save_interval is not None and save_dir is not None:
                if (step + 1) % self.save_interval == 0:
                    # 获取当前估计的高分辨率图像 x_start
                    save_tensor = x_start
                    
                    # 转换维度并放入CPU进行后续处理
                    save_img = save_tensor[0].cpu().numpy().transpose(1, 2, 0)
                    
                    # 处理归一化
                    if normalize_mode == 1:
                        save_img = save_img * normalize_scale
                    elif normalize_mode == 2:
                        save_img = (save_img + 1.0) / 2.0 * normalize_scale
                        
                    # 截断数据
                    save_img = np.clip(save_img, 0, normalize_scale)
                    
                    # 数据类型转换
                    if normalize_scale == 255:
                        save_img = save_img.astype(np.uint8)
                    elif normalize_scale == 1:
                        save_img = save_img.astype(np.float32)
                    elif normalize_scale == 10000:
                        save_img = save_img.astype(np.uint16)
                        
                    save_dir_path = Path(save_dir)
                    save_dir_path.mkdir(parents=True, exist_ok=True)
                    
                    save_name = f"{save_name_prefix}_step{step+1}.tif"
                    save_img_path = save_dir_path / save_name
                    
                    tifffile.imwrite(save_img_path, save_img)
            # -------------------------------

        return noisy_fine_img_02

    @torch.no_grad()
    def sample(
        self, 
        coarse_img_01, 
        coarse_img_02, 
        fine_img_01, 
        normalize_scale=10000, 
        normalize_mode=2, 
        save_name_prefix='ddim_mid'
    ):
        noisy_fine_img_02 = torch.randn_like(fine_img_01)
        if not self.is_ddim_sampling:
            print(f"Sample fun is DDPM")
            return self.p_sample_loop(
                coarse_img_01,
                coarse_img_02,
                fine_img_01,
                noisy_fine_img_02,
                save_dir=self.save_dir,
                save_interval=self.save_interval,
                normalize_scale=normalize_scale,
                normalize_mode=normalize_mode,
                save_name_prefix=save_name_prefix
            )
        else:
            print(f"Sample fun is DDIM with eta {self.ddim_sampling_eta}")
            return self.ddim_sample(
                coarse_img_01,
                coarse_img_02,
                fine_img_01,
                noisy_fine_img_02,
                clip_denoised=True,
                save_dir=self.save_dir,
                normalize_scale=normalize_scale,
                normalize_mode=normalize_mode,
                save_name_prefix=save_name_prefix
            )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(
        self, coarse_img_01, coarse_img_02, fine_img_01, fine_img_02, t, noise=None
    ):
        b, c, h, w = fine_img_02.shape
        noise = default(noise, lambda: torch.randn_like(fine_img_02))

        # noise sample

        noisy_fine_img_02 = self.q_sample(x_start=fine_img_02, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        # x_self_cond = None
        # if self.self_condition and random() < 0.5:
        #     with torch.no_grad():
        #         x_self_cond = self.model_predictions(x, t).pred_x_start
        #         x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(
            coarse_img_01, coarse_img_02, fine_img_01, noisy_fine_img_02, t
        )

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = fine_img_02
        else:
            raise ValueError(f'unknown objective {self.objective}')

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
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # img = normalize_to_neg_one_to_one(img)
        # 返回值是损失
        return self.p_losses(coarse_img_01, coarse_img_02, fine_img_01, fine_img_02, t)


# dataset classes
