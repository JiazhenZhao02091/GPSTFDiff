import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from diffusers import EDMEulerScheduler, EDMDPMSolverMultistepScheduler, DDIMScheduler
import tqdm
import math
import cv2
from pathlib import Path
from skimage import io

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
        sampling_timesteps = 100,
        sample_T1 = 15,
        sample_T2 = 50,
        T1 = 200,
        T2 = 500,
        alpha_mode = 'cosine',
    ):
        super().__init__()
        self.T1 = T1
        self.T2 = T2
        self.sample_T1 = sample_T1
        self.sample_T2 = sample_T2
        self.alpha_mode = alpha_mode
        print(f"alpha mode is {self.alpha_mode}")
        print(f"train_timesteps : {num_train_timesteps}")
        print(f"T1: {self.T1}, T2: {self.T2}")
        print(f"sample_timesteps: {sampling_timesteps}")
        print(f"sample_T1: {self.sample_T1}, sample_T2: {self.sample_T2}")

        self.model = model
        self.image_size = image_size
        self.loss_type = loss_type
        self.num_train_timesteps = num_train_timesteps
        self.sampling_timesteps = sampling_timesteps
        self.init_para(timesteps=num_train_timesteps)

        self.save_dir = "results/debug/tmp_6"
        self.current_iter = 0


    def laplacian_pyramid(self, img, levels=3):
        pyramid = []
        for i in range(levels):

            if i == levels - 1:
                pyramid.append(img)
                break
            down = F.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=False)
            up = F.interpolate(down, size=img.shape[2:], mode='bilinear', align_corners=False)
            pyramid.append(img - up)
            img = down
        pyramid.append(img)

        return pyramid[0], pyramid[1], pyramid[2]


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

    def generate_decay_tensors_linear(self, t1: int, t2: int, total_steps: int = 1000, initial_value: float = 1.0):
        time_steps = torch.arange(total_steps, dtype=torch.float32)

        def get_decay(duration):

            progress = time_steps / float(duration)

            values = initial_value * (1 - progress)

            values = torch.clamp(values, min=0.0)
            return values

        list1 = get_decay(total_steps)
        list2 = get_decay(t1)
        list3 = get_decay(t2)

        return list1, list2, list3

    def generate_decay_tensors_poly(self, t1: int, t2: int, total_steps: int = 1000, initial_value: float = 1.0, power: float = 2.0):
        time_steps = torch.arange(total_steps, dtype=torch.float32)

        def get_decay(duration):
            progress = time_steps / float(duration)

            progress = torch.clamp(progress, max=1.0)
            values = initial_value * torch.pow((1 - progress), power)

            mask = time_steps >= duration
            values[mask] = 0.0
            return values

        list1 = get_decay(total_steps)
        list2 = get_decay(t1)
        list3 = get_decay(t2)

        return list1, list2, list3


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


        if self.alpha_mode == 'cosine':
            alpha_3, alpha_1, alpha_2 = self.generate_decay_tensors_cosine(t1 = self.T1, t2 = self.T2, total_steps = timesteps)
        elif self.alpha_mode == 'linear':
            alpha_3, alpha_1, alpha_2 = self.generate_decay_tensors_linear(t1 = self.T1, t2 = self.T2, total_steps = timesteps)
        elif self.alpha_mode == 'poly':
            alpha_3, alpha_1, alpha_2 = self.generate_decay_tensors_poly(t1 = self.T1, t2 = self.T2, total_steps = timesteps)
        else:
            raise ValueError(f'unknown alpha mode {self.alpha_mode}')

        register_buffer('alpha_1', alpha_1)
        register_buffer('alpha_2', alpha_2)
        register_buffer('alpha_3', alpha_3)
        print(f"alpha_3[400]: {alpha_3[400]}, alpha_1[400]: {alpha_1[400]}, alpha_2[400]: {alpha_2[400]}")
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)

        register_buffer(
            'sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod)
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

    def get_xt(self, x0_t, t, scale=1):

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

        self.current_iter = self.current_iter + 1


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

        t = torch.randint(0, self.num_train_timesteps, (1,), device=device).long()
        t = t[0]

        if t < self.T1:
            t = torch.randint(0, self.T1, (b,), device=device).long()
            self.mode = 'x1+x2+x3'
        elif t < self.T2:
            t = torch.randint(self.T1, self.T2, (b,), device=device).long()
            self.mode = 'x2+x3'
        else:
            t = torch.randint(self.T2, self.num_train_timesteps, (b,), device=device).long()
            self.mode = 'x3'


        return self.p_losses(coarse_img_01, coarse_img_02, fine_img_01, fine_img_02, t)

    @torch.no_grad()
    def p_sample_loop(
        self, coarse_img_01, coarse_img_02, fine_img_01, noisy_fine_img_02
    ):
        xt = noisy_fine_img_02
        for t in tqdm.tqdm(
            reversed(range(0, self.sampling_timesteps)),
            desc='sampling loop time step',
            total=self.sampling_timesteps,
        ):

            batched_times = torch.full(
                    (noisy_fine_img_02.shape[0],),
                    t,
                    device=noisy_fine_img_02.device,
                    dtype=torch.long,
            )

            if t >= self.sample_T2:
                self.mode = 'x3'
                condition = self.get_condition(coarse_img_01, coarse_img_02, fine_img_01)
                x0_t = self.model(
                    xt,
                    batched_times,
                    condition,
                )
                if t == self.sample_T2:
                    x0_t = F.interpolate(x0_t, scale_factor=2, mode='bilinear', align_corners=False)
                    xt = self.get_xt(x0_t, batched_times, scale=2)
                else:
                    xt = self.get_xt(x0_t, batched_times)
            elif t >= self.sample_T1:
                self.mode = 'x2+x3'
                condition = self.get_condition(coarse_img_01, coarse_img_02, fine_img_01)
                x0_t = self.model(
                    xt,
                    batched_times,
                    condition,
                )
                if t == self.sample_T1:
                    x0_t = F.interpolate(x0_t, scale_factor=2, mode='bilinear', align_corners=False)
                    xt = self.get_xt(x0_t, batched_times, scale=2)
                else:
                    xt = self.get_xt(x0_t, batched_times)
            else:
                self.mode = 'x1+x2+x3'
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
        b,c,h,w = fine_img_01.shape
        noisy_fine_img_02 = torch.randn(b,c,h//4, w//4).to(fine_img_01.device)
        return self.p_sample_loop(
            coarse_img_01,
            coarse_img_02,
            fine_img_01,
            noisy_fine_img_02,
        )

    def save_prediction_grid(
        self,
        outputs: torch.Tensor,
        x0_t: torch.Tensor,
        iter_num: int,
        save_dir: str,
        normalize_mode: int = 2,
        img_interval: int = 10,
    ):
        if outputs.shape != x0_t.shape:
            raise ValueError("Outputs and x0_t must have the same shape.")

        B, C, H, W = outputs.shape


        h_num = 2

        w_num = B


        grid_height = (H + img_interval) * h_num + img_interval
        grid_width = (W + img_interval) * w_num + img_interval

        show_img = np.zeros(
            (grid_height, grid_width, 3),
            dtype=np.uint8
        )


        tensors_to_show = [x0_t, outputs]

        for h_index, tensor_batch in enumerate(tensors_to_show):
            for w_index in range(w_num):

                show_sub_img = tensor_batch[w_index].detach().cpu().numpy().transpose(1, 2, 0)


                if normalize_mode == 1:

                    show_sub_img = show_sub_img * 255.0
                elif normalize_mode == 2:

                    show_sub_img = (show_sub_img + 1.0) / 2.0 * 255.0


                if C == 6:

                    show_sub_img = show_sub_img[:, :, (3, 2, 1)]


                show_sub_img = np.clip(show_sub_img, 0, 255).astype(np.uint8)


                show_sub_img = cv2.resize(
                    show_sub_img, (W, H), interpolation=cv2.INTER_NEAREST
                )


                h_start = img_interval * (h_index + 1) + h_index * H
                h_end = h_start + H
                w_start = img_interval * (w_index + 1) + w_index * W
                w_end = w_start + W

                show_img[h_start:h_end, w_start:w_end, :] = show_sub_img


        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)


        show_name = f"iteration_{iter_num + 1:06d}.png"
        show_img_path = save_path / show_name

        io.imsave(str(show_img_path), show_img)
