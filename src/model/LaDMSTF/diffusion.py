from _typeshed import Self
from test.args import tmp
from STF.debug.test import timesteps
from STF.src.model.UViT.debug import batch_size
import torch
import numpy as np
import torch.nn as nn
from diffusers import EDMEulerScheduler


class EDMDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        num_train_timesteps=1000,         # 训练时从 0 - 999 中随机采样时间步t，噪声添加的精细粒度
        sampling_timesteps=None,
        loss_type='l1',
        prediction_type='sample',
    ):
        self.model = model
        self.image_size = image_size
        self.sheduler = EDMEulerScheduler(
            num_train_timesteps=num_train_timesteps,
            prediction_type = prediction_type
        )
        self.downsample = nn.Sequential(
            nn.AvgPool2d(2),
        )
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
        )

    def init_alpha_time_reduce(self):
        self.alpha_1 = []
        self.alpha_2 = []
        self.alpha_3 = []
        for i in range(self.sheduler.num_train_timesteps):  # 默认为 1000
            self.alpha_1.append(1 - i / 250 if i < 250 else 0)
            self.alpha_2.append(1 - i / 500 if i < 500 else 0)
            self.alpha_3.append(1 - i / 1000 if i < 1000 else 0)
        
        self.t2 = 500
        self.t1 = 250
        print("alpha init successful !!!")
        pass
    
    
    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def sample(self, coarse_img_01, coarse_img_02, fine_img_01):
        "采样一个噪声"
        pass

    # Forward  Noising 
    def _add(self, x1, x2, x3, t):
        if t > self.t2:
            "此时只有x3有值"
            return self.alpha_3[t] * x3
        elif t >self.t1 and t < self.t2:
            "此时为 x2 和 x3"
            x3_upsample = self.upsample(self.alpha_3[t] * x3)
            return x3_upsample + self.alpha_2[t] * x2
        else:
            "此时所有都可以"
            x3_upsample = self.upsample(self.alpha_3[t] * x3)
            x2_upsample = self.upsample(x3_upsample + self.alpha_2[t] * x2)
            return self.alpha_2[t] * x1 + x2_upsample

    def _sub(self, x):
        x3 = self.downsample(self.downsample(x))
        x2 = self.downsample(x) - self.upsample(x3)
        x1 = x - self.upsample(x2)
        
        return x1, x2, x3

    def get_noise_image(self, x, t):
        # R = 256, r = 128
        # noise = R / r * noise

        x1, x2 ,x3 = self._sub(x)
        x_sum = self._add(x1, x2, x3, t)
        noise = torch.randn_like(x_sum) 

        noise_image = self.scheduler.add_noise(x_sum, noise, t)
        return noise_image, noise

    def p_losses(self, coarse_img_01, coarse_img_02, fine_img_01, fine_img_02, t):
        "计算损失"
        tmp_noise = self.downsample(self.downsample(fine_img_02))
        noise = torch.randn_like(tmp_noise)
        batch_size = tmp_noise.shape[0]
        timestep = torch.randint(0, self.scheduler.num_train_timesteps, (batch_size,), device=fine_img_02.device)
        # noise_image = self.scheduler.add_noise(fine_img_02, noise, timestep)

        noise_image, noise = get_noise_image(fine_img_02, timestep)

        outputs = self.model(coarse_img_01, coarse_img_02, fine_img_01, noise_image, timestep)
        
        if self.prediction_type == 'sample':
            loss = self.loss_fn(outputs, noise_image)
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
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # img = normalize_to_neg_one_to_one(img)
        # 返回值是损失

        return self.p_losses(coarse_img_01, coarse_img_02, fine_img_01, fine_img_02, t)


    @torch.no_grad()
    def sample(self, coarse_img_01, coarse_img_02, fine_img_01):
        low_tmp = self.downsample(self.downsample(fine_img_01))
        noisy_image = torch.randn_like(low_tmp)

        self.scheduler.set_timesteps(self.sampling_timesteps)

        # Sample Loop
        for i,t in enumerate(self.scheduler.timesteps):
            timesteps = torch.full((batch_size,), t, device=fine_img_01.device, dtype=torch.long)
            noisy_image = self.scheduler.scale_model_input(noisy_image, timesteps)
            model_output = self.model(coarse_img_01, coarse_img_02, fine_img_01, noisy_image, timesteps)

            noisy_image = self.scheduler.step(model_output, timesteps, noisy_image).prev_sample

            if i == 50:
                noisy_image = self.upsample(noisy_image)
                noise = torch.randn_like(noisy_image)
                # x = upsample(x) + ∂ * R/r * epsilon
                # timesteps get sigma
                noisy_image = self.scheduler.add_noise(noisy_image, 2 * noise, timesteps) 
                print(f"noisy image shape is {noisy_image.shape}")
            if i == 80:
                noisy_image = self.upsample(noisy_image)
                noise = torch.randn_like(noisy_image)
                # x = upsample(x) + ∂ * R/r * epsilon
                # timesteps get sigma
                noisy_image = self.scheduler.add_noise(noisy_image, 2 * noise, timesteps) 
                print(f"noisy image shape is {noisy_image.shape}")
        
        return noisy_image