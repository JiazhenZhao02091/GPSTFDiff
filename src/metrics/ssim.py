# Copyright (c) OpenMMLab. All rights reserved.
import statistics
from torch import nn
import torch
import cv2
import torch.nn.functional as F


class StructuralSimilarity(nn.Module):
    """Peak Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Metrics:
        - PSNR (float): Peak Signal-to-Noise Ratio
    """

    __name__ = 'SSIM'

    def __init__(
        self,
        gussian_kernel_size=11,
        gussian_sigma=1.5,
        data_range=255,
        K1=0.01,
        K2=0.03,
        is_reduce_channel=True,
    ):
        super().__init__()
        self.gussian_kernel = self._get_gussian_kernel(
            gussian_kernel_size, gussian_sigma
        )
        self.C1 = (K1 * data_range) ** 2
        self.C2 = (K2 * data_range) ** 2
        # self.unfold = nn.Unfold(gussian_kernel_size) # 移除未使用的属性
        self.gussian_kernel_size = gussian_kernel_size
        self.is_reduce_channel = is_reduce_channel

    def _get_gussian_kernel(self, gussian_kernel_size, gussian_sigma):
        d_kernel = cv2.getGaussianKernel(gussian_kernel_size, gussian_sigma)
        _2d_kernel = d_kernel @ d_kernel.T
        kernel = torch.FloatTensor(_2d_kernel).unsqueeze(0).unsqueeze(0)
        kernel = nn.Parameter(kernel, requires_grad=False)
        return kernel

    def forward(self, gt, pred, mask=None):
        """Process an image.

        Args:
            BCHW format.
            gt (Torch | np.ndarray): GT image.
            pred (Torch | np.ndarray): Pred image.
            mask (Torch | np.ndarray): Mask of evaluation.
        Returns:
            np.ndarray: SSIM result.
        """
        assert gt.shape == pred.shape
        b, c, h, w = gt.shape

        kernel = self.gussian_kernel.repeat_interleave(repeats=c, dim=0)

        # 1. Mask 预处理
        if mask is not None:
             # 扩展 mask 以匹配 (B, C, H, W)
             if mask.dim() == 3:
                mask = mask.unsqueeze(1)
             # 有些 mask 可能是 1 通道，需要扩展成 C 通道以匹配 group convolution
             if mask.shape[1] == 1:
                mask = mask.repeat(1, c, 1, 1)
             
             mask = (mask > 0).float()
             
             # 将无效区域置零
             gt = gt * mask
             pred = pred * mask

        # 2. 基础卷积 (Sums)
        # 包含了: sum(gt), sum(pred), sum(gt^2), sum(pred^2), sum(gt*pred)
        cube = torch.cat((gt, pred, gt * gt, pred * pred, gt * pred), dim=0)
        # 注意：F.conv2d 默认无 padding，输出尺寸会变小 (h-k+1, w-k+1)
        statistic_cube = F.conv2d(cube, weight=kernel, groups=c)

        # 解包 (此时得到的只是加权和，还不是均值)
        mu_gt_sum = statistic_cube[:b]
        mu_pred_sum = statistic_cube[b : 2 * b]
        sq_gt_sum = statistic_cube[2 * b : 3 * b]
        sq_pred_sum = statistic_cube[3 * b : 4 * b]
        mul_sum = statistic_cube[4 * b :]

        # 3. 计算归一化因子 W 并计算统计量
        if mask is not None:
             # 计算卷积窗口内的有效权重和 W
             # 这步是关键：计算每个窗口里到底有多少个“有效像素”
             W = F.conv2d(mask, weight=kernel, groups=c)
             
             # 【稳定性保护 1】 防止除零
             # 如果 W 太小，说明窗口内没几个有效像素，计算出来的均值方差不可信
             W_safe = torch.clamp(W, min=1e-4) 

             # 计算正确的局部均值 (Normalized Mean)
             mu_gt = mu_gt_sum / W_safe
             mu_pred = mu_pred_sum / W_safe
             
             # 计算正确的局部方差/协方差 (利用公式 Var(X) = E[X^2] - (E[X])^2)
             sigma_gt_sq = (sq_gt_sum / W_safe) - mu_gt ** 2
             sigma_pred_sq = (sq_pred_sum / W_safe) - mu_pred ** 2
             cor_gt_pred = (mul_sum / W_safe) - mu_gt * mu_pred
             
             # 【稳定性保护 2】 由于数值精度问题，方差可能出现微小的负数，clamp回0
             sigma_gt_sq = torch.clamp(sigma_gt_sq, min=0)
             sigma_pred_sq = torch.clamp(sigma_pred_sq, min=0)

             # 对齐 Mask 尺寸 (因为卷积让图变小了，我们需要取 Mask 的中间部分)
             # kernel_size 比如是 11，padding 就是 5
             pad = self.gussian_kernel_size // 2
             # 取中心有效区域作为最终的 valid_mask
             valid_mask = mask[:, :, pad : h - pad, pad : w - pad]
             
             # 【稳定性保护 3】 只有当窗口内有超过一半是有效像素时，才统计该点的 SSIM
             # 这里 kernel 是归一化的，满窗口 W 接近 1.0。我们要求 W > 0.5
             valid_mask = valid_mask * (W > 0.5).float()

        else:
             # 无 Mask 时，假设 W=1 (核已经归一化)
             mu_gt = mu_gt_sum
             mu_pred = mu_pred_sum
             
             sigma_gt_sq = statistic_cube[2 * b : 3 * b] - mu_gt ** 2
             sigma_pred_sq = statistic_cube[3 * b : 4 * b] - mu_pred ** 2
             cor_gt_pred = statistic_cube[4 * b :] - mu_gt * mu_pred

        # 4. SSIM 公式计算
        # 重构这部分以匹配变量名
        mu_gt_sq = mu_gt ** 2
        mu_pred_sq = mu_pred ** 2
        mu_gt_mul_pred = mu_gt * mu_pred

        cs_map = (2 * cor_gt_pred + self.C2) / (
            sigma_gt_sq + sigma_pred_sq + self.C2
        )
        
        ssim_map = (
            (2 * mu_gt_mul_pred + self.C1) / (mu_gt_sq + mu_pred_sq + self.C1)
        ) * cs_map

        # 5. 返回结果
        if mask is not None:
             # 再次确保只计算有效区域
             ssim_map = ssim_map * valid_mask
             valid_count = torch.sum(valid_mask, dim=(-2, -1))
             
             # 防止 valid_count 为 0
             valid_count = torch.clamp(valid_count, min=1.0)
             
             ssim_per_channel = torch.sum(ssim_map, dim=(-2, -1)) / valid_count
             
             if self.is_reduce_channel:
                 return ssim_per_channel.mean()
             else:
                 return ssim_per_channel.mean(0) 

        if self.is_reduce_channel:
            return torch.mean(ssim_map)
        else:
            return torch.mean(ssim_map, dim=(0, 2, 3))


SSIM = StructuralSimilarity

# python -m src.metrics.ssim
if __name__ == '__main__':
    # 基础测试
    preds = torch.rand([16, 3, 256, 256], generator=torch.manual_seed(42)).to('cpu')
    target = torch.rand([16, 3, 256, 256], generator=torch.manual_seed(123)).to('cpu')

    ssim = SSIM(data_range=1.0).to('cpu')
    a = ssim(target, preds)
    print("Normal SSIM:", a)

    # Mask 测试
    print("-" * 20)
    print("Mask Test:")
    
    # 构造两个完全一样的图像，SSIM 应该是 1.0
    img1 = torch.ones([1, 1, 64, 64])
    img2 = torch.ones([1, 1, 64, 64])
    
    # 构造一个 Mask，仅中间区域有效
    mask = torch.zeros([1, 1, 64, 64])
    mask[:, :, 16:48, 16:48] = 1

    # 在 Mask 区域外制造巨大噪声
    img2_noisy = img2.clone()
    img2_noisy[:, :, :10, :10] = 100.0 # 这些区域 Mask 是 0
    
    val_with_mask = ssim(img1, img2_noisy, mask)
    print(f"Same content inside mask, Noise outside. SSIM with mask: {val_with_mask.item():.4f}")
    # 预期: 接近 1.0，因为噪声区被 mask 过滤掉了

    val_without_mask = ssim(img1, img2_noisy)
    print(f"Same content inside mask, Noise outside. SSIM without mask: {val_without_mask.item():.4f}")
    # 预期: 远小于 1.0，因为噪声区拉低了分数
