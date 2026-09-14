# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn
import torch


# class PeakSignalNoiseRatio(nn.Module):
#     """Peak Signal-to-Noise Ratio.

#     Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

#     Metrics:
#         - PSNR (float): Peak Signal-to-Noise Ratio
#     """

#     __name__ = 'PSNR'

#     def __init__(self, max_value=255):
#         super().__init__()
#         self.max_value = max_value

#     def forward(self, gt, pred):
#         """Process an image.

#         Args:
#             gt (Torch | np.ndarray): GT image.
#             pred (Torch | np.ndarray): Pred image.
#             mask (Torch | np.ndarray): Mask of evaluation.
#         Returns:
#             np.ndarray: PSNR result.
#         """
#         mse_value = ((gt - pred) ** 2).mean()
#         if mse_value == 0:
#             result = torch.tensor(torch.inf)
#         else:
#             result = 20.0 * torch.log10(self.max_value / torch.sqrt(mse_value))

#         return result

class PeakSignalNoiseRatio(nn.Module):

    __name__ = 'PSNRONE'

    def __init__(self, max_value=255, is_reduce_channel=True): # 1. 添加参数
        super().__init__()
        self.max_value = max_value
        self.is_reduce_channel = is_reduce_channel # 2. 保存参数

    def forward(self, gt, pred):
        # 3. 修改均值计算逻辑
        if self.is_reduce_channel:
            mse_value = ((gt - pred) ** 2).mean()
        else:
            # 假设输入是 (B, C, H, W)，在 (B, H, W) 上求平均，保留 C
            mse_value = ((gt - pred) ** 2).mean(dim=(0, 2, 3)) 

        if self.is_reduce_channel:
            if mse_value == 0:
                result = torch.tensor(torch.inf)
            else:
                result = 20.0 * torch.log10(self.max_value / torch.sqrt(mse_value))
        else:
            # 针对每个通道分别计算 log
            result = 20.0 * torch.log10(self.max_value / torch.sqrt(mse_value))
            
        return result


PSNRONE = PeakSignalNoiseRatio
