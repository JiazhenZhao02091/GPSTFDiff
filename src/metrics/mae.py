# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn
import torch


class MeanAbsoluteError(nn.Module):
    """Peak Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Metrics:
        - PSNR (float): Peak Signal-to-Noise Ratio
    """

    __name__ = 'MAE'

    def __init__(self, is_reduce_channel=True):
        super().__init__()
        self.is_reduce_channel = is_reduce_channel  # True返回平均值

    def forward(self, gt, pred, mask=None):
        """Process an image.

        Args:
            gt (Torch | np.ndarray): GT image.
            pred (Torch | np.ndarray): Pred image.
            mask (Torch | np.ndarray): Mask of evaluation.
        Returns:
            np.ndarray: MAE result.
        """
        if mask is not None:
            # 扩展 mask 维度以匹配输入 (假设 mask 是 Bx1xHxW 或 BxCxHxW)
            if mask.dim() == 3: 
                mask = mask.unsqueeze(1)
            if mask.shape[1] == 1 and gt.shape[1] != 1:
                mask = mask.expand(-1, gt.shape[1], -1, -1)
            
            # 确保 mask 为布尔类型或 0/1
            valid_mask = mask > 0

            # 只选取有效像素
            diff = torch.abs(gt[valid_mask] - pred[valid_mask])
            
            # 如果需要保留通道信息 (is_reduce_channel=False)，逻辑会很复杂
            # 因为 mask 后每个通道的有效像素数量可能不同，无法保持 Tensor 形状。
            # 这里通常仅支持返回标量平均值
            return diff.mean()
        
        if self.is_reduce_channel:
            result = (torch.abs(gt - pred)).mean()
        else:
            result = torch.mean(torch.abs(gt - pred), dim=(0, 2, 3))
        # result = (torch.abs(gt - pred)).mean()
        return result


MAE = MeanAbsoluteError
