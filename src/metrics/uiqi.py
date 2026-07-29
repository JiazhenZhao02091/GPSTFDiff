# Correlation Coefficient
import statistics
from torch import nn
import torch
import cv2
import torch.nn.functional as F

epsilon = 1e-10

class UniversalImageQualityIndex(nn.Module):
    """Peak Signal-to-Noise Ratio.
    """

    __name__ = 'UIQI'

    def __init__(
        self,
        # gussian_kernel_size=11,
        # gussian_sigma=1.5,
        is_reduce_channel=True,
    ):
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
            np.ndarray: UIQI result.
        """
        assert gt.shape == pred.shape
        b, c, h, w = gt.shape

        if mask is not None:
            # 1. 预处理 Mask
            if mask.dim() == 3:
                mask = mask.unsqueeze(1) # B, 1, H, W
            mask = (mask > 0).float()
            
            # 计算有效像素个数 N (B, C, 1, 1)
            N = torch.sum(mask, dim=(-2, -1), keepdim=True)
            # 防止除以0
            N = torch.clamp(N, min=1.0)

            # 过滤数据 (无效区域置0)
            gt = gt * mask
            pred = pred * mask

            # 2. 计算带 Mask 的均值 (Sum / N)
            mu_gt = torch.sum(gt, dim=(-2, -1), keepdim=True) / N
            mu_pred = torch.sum(pred, dim=(-2, -1), keepdim=True) / N

        else:
            # 原始无 Mask 逻辑
            mu_gt = torch.mean(gt, dim=(-2, -1), keepdim=True)
            mu_pred = torch.mean(pred, dim=(-2, -1), keepdim=True)
        
        # 3. 准备计算方差和协方差的中间变量
        mu_gt_sq = mu_gt**2
        mu_pred_sq = mu_pred**2
        mu_gt_mul_mu_pred = mu_gt * mu_pred

        if mask is not None:
            # 带 Mask 的方差: (Sum(x^2) / N) - mu^2
            # 注意：这里 gt*gt 的无效区域已经是0了，所以 sum 没问题
            sigma_gt_sq = torch.sum(gt**2, dim=(-2, -1), keepdim=True) / N - mu_gt_sq
            sigma_pred_sq = torch.sum(pred**2, dim=(-2, -1), keepdim=True) / N - mu_pred_sq
            
            # 带 Mask 的协方差: (Sum(xy) / N) - mu_x*mu_y
            cor_gt_pred = torch.sum(gt * pred, dim=(-2, -1), keepdim=True) / N - mu_gt_mul_mu_pred
        else:
            sigma_gt_sq = torch.mean(gt**2, dim=(-2, -1), keepdim=True) - mu_gt_sq
            sigma_pred_sq = torch.mean(pred**2, dim=(-2, -1), keepdim=True) - mu_pred_sq
            cor_gt_pred = torch.mean(gt * pred, dim=(-2, -1), keepdim=True) - mu_gt_mul_mu_pred

        # 4. 计算 UIQI map
        uiqi_map = (
            4
            * cor_gt_pred
            * mu_gt_mul_mu_pred
            / (
                (sigma_gt_sq + sigma_pred_sq + epsilon)
                * (mu_gt_sq + mu_pred_sq + epsilon)
            )
        )

        # 处理可能出现的 NaN (当某张图完全被 Mask 掉时)
        uiqi_map = torch.nan_to_num(uiqi_map, 0.0)

        if self.is_reduce_channel:
            return torch.mean(uiqi_map)
        else:
            return torch.mean(uiqi_map, dim=(0, -2, -1))


UIQI = UniversalImageQualityIndex

# python -m src.metrics.uiqi
if __name__ == '__main__':
    # 测试代码更新以验证 Mask
    preds = torch.ones([1, 1, 10, 10]).to('cpu') 
    target = torch.ones([1, 1, 10, 10]).to('cpu')
    
    # 造一个 Mask，只有左上角 2x2 是有效的
    mask = torch.zeros([1, 1, 10, 10]).to('cpu')
    mask[:, :, :2, :2] = 1

    uiqi = UIQI()
    
    # 情况1: 不传 mask (背景虽然是0，但参与分母计算，且均值不再是 1)
    # 期望: 远小于 1.0 (因为均值被拉低了)
    # 注意：为了模拟效果，输入数据本身在 mask 外要是 0
    preds_masked = preds * mask
    target_masked = target * mask
    print("Without Mask Logic (Input Zeroed):", uiqi(target_masked, preds_masked))
    
    # 情况2: 传 mask
    # 期望: 接近 1.0 (因为只计算有效区域)
    print("With Mask Logic:", uiqi(target_masked, preds_masked, mask))
