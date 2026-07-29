# Correlation Coefficient
import statistics
from torch import nn
import torch
import cv2
import torch.nn.functional as F


class CorrelationCoefficient(nn.Module):
    """Peak Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Metrics:
        - PSNR (float): Peak Signal-to-Noise Ratio
    """

    __name__ = 'CC'

    def __init__(self, is_reduce_channel=True):
        super().__init__()
        self.is_reduce_channel = is_reduce_channel

    def forward(self, gt, pred, mask=None):
        """Process an image.

        Args:
            BCHW format.
            gt (Torch | np.ndarray): GT image.
            pred (Torch | np.ndarray): Pred image.
            mask (Torch | np.ndarray): Mask of evaluation.
        Returns:
            np.ndarray: PSNR result.
        """
        assert gt.shape == pred.shape
        b, c, h, w = gt.shape

        if mask is not None:
            # 扩展 mask 以匹配 (B, C, H, W)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            # 确保 mask 是浮点型以便做乘法 (0.0 或 1.0)
            mask = (mask > 0).float()
            
            # 计算每个样本每个通道的有效像素数 N
            # Sum over H, W
            N = torch.sum(mask, dim=(-2, -1))
            
            # 使用 mask 过滤数据
            gt = gt * mask
            pred = pred * mask

            # 计算均值 Sum(X) / N
            mu_gt = torch.sum(gt, dim=(-2, -1)) / N
            mu_pred = torch.sum(pred, dim=(-2, -1)) / N
            
            # 计算协方差 E[XY] - E[X]E[Y]
            # Sum(X*Y)/N
            mu_gt_pred = torch.sum(gt * pred, dim=(-2, -1)) / N
            cor_gt_pred = mu_gt_pred - mu_gt * mu_pred
            
            # 计算方差 E[X^2] - (E[X])^2
            var_gt = torch.sum(gt * gt, dim=(-2, -1)) / N - mu_gt * mu_gt
            var_pred = torch.sum(pred * pred, dim=(-2, -1)) / N - mu_pred * mu_pred

            cc_per_channel = cor_gt_pred / (var_gt * var_pred).sqrt()
            
            # 处理可能的 NaN (如果某通道全是 mask 0)
            cc_per_channel = torch.nan_to_num(cc_per_channel, 0.0)

            if self.is_reduce_channel:
                return cc_per_channel.mean()
            else:
                return cc_per_channel.mean(0)

        # 原始未 Mask 的逻辑
        mu_gt = torch.mean(gt, dim=(-2, -1))
        mu_pred = torch.mean(pred, dim=(-2, -1))
        mu_gt_pred = torch.mean(gt * pred, dim=(-2, -1))

        cor_gt_pred = mu_gt_pred - mu_gt * mu_pred

        # cor_gt_pred = torch.mean((gt - mu_gt) * (pred - mu_pred), dim=(-2, -1))

        var_gt = torch.mean(gt * gt, dim=(-2, -1)) - mu_gt * mu_gt
        var_pred = torch.mean(input=pred * pred, dim=(-2, -1)) - mu_pred * mu_pred

        cc_per_channel = cor_gt_pred / (var_gt * var_pred).sqrt()

        if self.is_reduce_channel:
            return cc_per_channel.mean()
        else:
            return cc_per_channel.mean(0)


CC = CorrelationCoefficient

# python -m src.metrics.cc
if __name__ == '__main__':
    preds = torch.rand([1, 1, 256, 2], generator=torch.manual_seed(42)).to('cpu')
    target = torch.rand([1, 1, 256, 2], generator=torch.manual_seed(123)).to('cpu')

    cc = CC()
    a = cc(target, preds)
    print(a)

    from torchmetrics import PearsonCorrCoef

    pearson = PearsonCorrCoef()
    b = pearson(preds.flatten(), target.flatten())
    print(b)
