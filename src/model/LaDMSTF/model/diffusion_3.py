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


class EDMDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        num_train_timesteps=1000,         # 训练时从 0 - 999 中随机采样时间步t，噪声添加的精细粒度
        loss_type='l1',
        prediction_type='sample',
        sampling_timesteps = 100,
    ):
        super().__init__()
        self.T1 = 100       # t1 interval
        self.T2 = 500       # t2 interval
        self.sample_T1 = (self.T1 * sampling_timesteps // num_train_timesteps) if self.T1 > 200 else 15
        self.sample_T2 = self.T2 * sampling_timesteps // num_train_timesteps 
        print(f"sample_timesteps: {sampling_timesteps}")
        print(f"sample_T1: {self.sample_T1}, sample_T2: {self.sample_T2}")

        self.model = model
        self.image_size = image_size
        self.loss_type = loss_type
        self.prediction_type = prediction_type  # Unuse
        self.num_train_timesteps = num_train_timesteps
        self.sampling_timesteps = sampling_timesteps
        self.init_para(timesteps=num_train_timesteps)

        self.save_dir = "/data/zhaojiazhen/STF/tmp/tmp_6"
        self.current_iter = 0

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
            down = F.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=False)
            up = F.interpolate(down, size=img.shape[2:], mode='bilinear', align_corners=False)
            pyramid.append(img - up)
            img = down
        pyramid.append(img)
        # print(f"pyramid 的长度 :{len(pyramid)}")
        return pyramid[0], pyramid[1], pyramid[2]
        
    " 参数 "
    def init_para(self, beta_schedule='cosine', timesteps=1000):
        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )
        " get betas, betas is a sequence "
        if beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        " get alphas "
        alphas = 1.0 - betas                            # alpha = 1 - beta
        alphas_cumprod = torch.cumprod(alphas, dim=0)   # alpha_cumprod = alpha_0 * alpha_1 * ... * alpha_t
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        " 衰减因子 "
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

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def get_x0_t(self, x1, x2, x3 , t):
        " alpha 是个list "
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
        # x0_t = x3
        # if 'x2' in self.mode:
        #     x0_t = F.interpolate(x0_t, scale_factor=2, mode='nearest')
        #     # x0_t = F.interpolate(x0_t, scale_factor=2, mode='bilinear', align_corners=False)
        #     x0_t = x0_t + x2
        # if 'x1' in self.mode:
        #     x0_t = F.interpolate(x0_t, scale_factor=2, mode='nearest')
        #     x0_t = x0_t + x1
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
        
        " 拉普拉斯分解 "
        x1, x2, x3 = self.laplacian_pyramid(fine_img_02, levels=3)  
        " 计算x0_t "
        x0_t = self.get_x0_t(x1, x2, x3, t)  
        " 计算x_t "
        x_t = self.get_xt(x0_t, t)
        condition = self.get_condition(coarse_img_01, coarse_img_02, fine_img_01)
        " 模型输出 "
        outputs = self.model(x_t, t, condition)
        # self.save_prediction_grid(
        #     outputs,
        #     x0_t,
        #     self.current_iter,
        #     self.save_dir,
        # )
        " 损失值 "
        loss = self.loss_fn(outputs, x0_t)  # prediction type is "sample"

        # return loss.mean()
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
        # t = torch.zeros(b,).to(device).long()
        " 返回损失值 "
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
            # timesteps
            batched_times = torch.full(
                    (noisy_fine_img_02.shape[0],),
                    t,
                    device=noisy_fine_img_02.device,
                    dtype=torch.long,
            )
            " sample if "
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
                    xt = self.get_xt(x0_t, batched_times, scale=2)  # xt = 
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
        """
            Saves a grid visualization of model outputs and ground truth for a batch.

            The grid format is 2 rows: 
            Row 1: Ground Truth (x0_t) images
            Row 2: Model Output (outputs) images

            Args:`
                outputs (torch.Tensor): Model output tensor (B, C, H, W).
                x0_t (torch.Tensor): Ground truth tensor (B, C, H, W).
                iter_num (int): Current iteration number, used for file naming.
                save_dir (str): The path to the directory where the image will be saved.
                normalize_mode (int): Denormalization mode (1: * 255.0, 2: (+ 1.0) / 2.0 * 255.0).
                img_interval (int): Pixel space between images in the grid.
        """
        if outputs.shape != x0_t.shape:
            raise ValueError("Outputs and x0_t must have the same shape.")

        B, C, H, W = outputs.shape
        
        # 2 rows (Ground Truth, Output)
        h_num = 2 
        # Batch size columns
        w_num = B
        
        # Calculate the size of the final grid image
        grid_height = (H + img_interval) * h_num + img_interval
        grid_width = (W + img_interval) * w_num + img_interval
        
        show_img = np.zeros(
            (grid_height, grid_width, 3), 
            dtype=np.uint8
        )

        # Combine tensors for easy iteration: Row 0 is x0_t, Row 1 is outputs
        tensors_to_show = [x0_t, outputs]

        for h_index, tensor_batch in enumerate(tensors_to_show):
            for w_index in range(w_num):
                # Select the sub-image from the batch
                show_sub_img = tensor_batch[w_index].detach().cpu().numpy().transpose(1, 2, 0)

                # --- Denormalization and Channel Handling ---
                if normalize_mode == 1:
                    # [0, 1] -> [0, 255]
                    show_sub_img = show_sub_img * 255.0
                elif normalize_mode == 2:
                    # [-1, 1] -> [0, 255]
                    show_sub_img = (show_sub_img + 1.0) / 2.0 * 255.0
                
                # Replicate the original channel swap logic if C=6
                if C == 6:
                    # Assuming C=6 means the 4th, 3rd, and 2nd channels are the desired RGB
                    show_sub_img = show_sub_img[:, :, (3, 2, 1)]
                
                # Clip and convert to unsigned 8-bit integer
                show_sub_img = np.clip(show_sub_img, 0, 255).astype(np.uint8)

                # Resize (even though they are likely the correct size, keep logic)
                # Use H, W from tensor shape
                show_sub_img = cv2.resize(
                    show_sub_img, (W, H), interpolation=cv2.INTER_NEAREST
                )
                
                # --- Grid Placement ---
                # Calculate coordinates for the current cell
                h_start = img_interval * (h_index + 1) + h_index * H
                h_end = h_start + H
                w_start = img_interval * (w_index + 1) + w_index * W
                w_end = w_start + W
                
                show_img[h_start:h_end, w_start:w_end, :] = show_sub_img

        # --- Saving ---
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Naming convention: iter + 1
        show_name = f"iteration_{iter_num + 1:06d}.png"
        show_img_path = save_path / show_name

        io.imsave(str(show_img_path), show_img)