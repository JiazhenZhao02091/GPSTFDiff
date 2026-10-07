import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import cv2
import numpy as np
import inspect
from functools import partial
from collections import namedtuple
from tqdm.auto import tqdm

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

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

def pyrdown_nchw(x):
    b, c, h, w = x.shape
    images = x.detach().cpu().float().reshape(b * c, h, w).numpy()
    down = np.stack([cv2.pyrDown(img) for img in images])
    return torch.from_numpy(down.reshape(b, c, *down.shape[-2:])).to(
        device=x.device, dtype=x.dtype
    )

def cosine_beta_schedule(timesteps, s=0.008):
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
        num_train_timesteps=1000,
        sampling_timesteps = 100,
        objective='pred_x0',
        mode = "x1+x2+x3",
        T1 = 100,
        T2 = 150,
        ddim_sampling_eta=1.0,
        beta_schedule = 'cosine',
        model_x3 = None,
        model_x2_x3 = None,
        model_x1_x2_x3 = None,
    ):
        super().__init__()

        self.mode = mode
        self.model_x3 = model_x3
        self.model_x2_x3 = model_x2_x3
        self.model_x1_x2_x3 = model_x1_x2_x3

        self.T1 = T1
        self.T2 = T2
        self.sample_T1 = T1 / num_train_timesteps * sampling_timesteps
        self.sample_T2 = T2 / num_train_timesteps * sampling_timesteps

        self.ddim_sampling_eta = ddim_sampling_eta
        self.model = model
        self.image_size = image_size
        self.num_train_timesteps = num_train_timesteps
        self.sampling_timesteps = sampling_timesteps
        self.num_timesteps = num_train_timesteps
        self.self_condition = False
        self.objective = objective
        self._condition_forward_cache = {}
        self.init_para(beta_schedule=beta_schedule)

    def init_para(self, beta_schedule='cosine'):
        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        if beta_schedule == 'cosine':
            betas = cosine_beta_schedule(self.num_train_timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

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

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def uses_condition_forward(self, model):
        model_id = id(model)
        if model_id not in self._condition_forward_cache:
            try:
                parameters = list(inspect.signature(model.forward).parameters)
            except (TypeError, ValueError):
                parameters = []

            if parameters and parameters[0] == "self":
                parameters = parameters[1:]

            self._condition_forward_cache[model_id] = (
                len(parameters) >= 3
                and parameters[0] == "noisy_fine_img_02"
                and parameters[1] in {"time", "t"}
            )

        return self._condition_forward_cache[model_id]

    def model_forward(
        self,
        model,
        coarse_img_01,
        coarse_img_02,
        fine_img_01,
        noisy_fine_img_02,
        t,
        x_self_cond=None,
    ):
        if self.uses_condition_forward(model):
            condition = self.get_condition(coarse_img_01, coarse_img_02, fine_img_01)
            return model(noisy_fine_img_02, t, condition, x_self_cond)

        return model(
            coarse_img_01,
            coarse_img_02,
            fine_img_01,
            noisy_fine_img_02,
            t,
            x_self_cond,
        )

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
        model_output = self.model_forward(
            self.model,
            coarse_img_01,
            coarse_img_02,
            fine_img_01,
            noisy_fine_img_02,
            t,
            x_self_cond,
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
            return condition
        if self.mode == 'x2+x3':
            return pyrdown_nchw(condition)
        if self.mode == 'x3':
            return pyrdown_nchw(pyrdown_nchw(condition))
        return condition

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
        noise = torch.randn_like(noisy_fine_img_02) if t > 0 else 0.0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(
        self, coarse_img_01, coarse_img_02, fine_img_01, noisy_fine_img_02
    ):
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
        )
        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )

        x_start = None
        i = sampling_timesteps

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

            i = i - 1
            if i == self.sample_T2:
                self.mode = "x2+x3"
                self.model = self.model_x2_x3
                scale = 1
                x_start_up = cv2_interpolate(x_start, scale_factor=2, mode='bilinear', align_corners=False)
                noise = torch.randn_like(x_start_up)

                noisy_fine_img_02 = x_start_up + noise * (1 - alpha_next).sqrt() * scale
                continue

            if i == self.sample_T1:
                self.mode = "x1+x2+x3"
                self.model = self.model_x1_x2_x3
                scale = 1
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
        self.mode = "x3"
        self.model = self.model_x3
        return self.ddim_sample(
            coarse_img_01,
            coarse_img_02,
            fine_img_01,
            noisy_fine_img_02,
        )
