from torch import nn
import torch
import torch.nn.functional as F
import cv2
from ._utils import get_valid_mask, masked_mean, prepare_images


class StructuralSimilarity(nn.Module):
    __name__ = 'SSIM'

    def __init__(self, gussian_kernel_size=11, gussian_sigma=1.5,
                 data_range=255, K1=0.01, K2=0.03, is_reduce_channel=True):
        super().__init__()
        if gussian_kernel_size < 1 or gussian_kernel_size % 2 == 0:
            raise ValueError("gussian_kernel_size must be a positive odd integer")
        if gussian_sigma <= 0 or data_range <= 0:
            raise ValueError("gussian_sigma and data_range must be positive")
        self.gussian_kernel = self._get_gussian_kernel(gussian_kernel_size, gussian_sigma)
        self.C1 = (K1 * data_range) ** 2
        self.C2 = (K2 * data_range) ** 2
        self.gussian_kernel_size = gussian_kernel_size
        self.is_reduce_channel = is_reduce_channel

    def _get_gussian_kernel(self, gussian_kernel_size, gussian_sigma):
        kernel = cv2.getGaussianKernel(gussian_kernel_size, gussian_sigma)
        kernel = torch.tensor(kernel @ kernel.T, dtype=torch.float32)[None, None]
        return nn.Parameter(kernel, requires_grad=False)

    def forward(self, gt, pred, mask=None):
        gt, pred = prepare_images(gt, pred)
        b, c, h, w = gt.shape
        if min(h, w) < self.gussian_kernel_size:
            raise ValueError("image dimensions must be at least gussian_kernel_size")
        kernel = self.gussian_kernel.to(device=gt.device, dtype=gt.dtype).expand(c, -1, -1, -1)
        valid = None if mask is None else get_valid_mask(mask, gt)
        if valid is not None:
            gt = torch.where(valid, gt, 0)
            pred = torch.where(valid, pred, 0)
        cube = torch.cat((gt, pred, gt.square(), pred.square(), gt * pred), dim=0)
        statistics = F.conv2d(cube, kernel, groups=c)
        mu_gt, mu_pred, sq_gt, sq_pred, product = statistics.split(b, dim=0)
        if valid is not None:
            weight = F.conv2d(valid.to(gt.dtype), kernel, groups=c)
            weight_safe = weight.clamp(min=torch.finfo(gt.dtype).tiny)
            mu_gt, mu_pred = mu_gt / weight_safe, mu_pred / weight_safe
            sq_gt, sq_pred, product = sq_gt / weight_safe, sq_pred / weight_safe, product / weight_safe
            pad = self.gussian_kernel_size // 2
            valid = valid[:, :, pad:pad + weight.shape[-2], pad:pad + weight.shape[-1]] & (weight > 0.5)
        var_gt = (sq_gt - mu_gt.square()).clamp(min=0)
        var_pred = (sq_pred - mu_pred.square()).clamp(min=0)
        covariance = product - mu_gt * mu_pred
        covariance_limit = (var_gt * var_pred).sqrt()
        covariance = torch.minimum(torch.maximum(covariance, -covariance_limit), covariance_limit)
        ssim_map = ((2 * mu_gt * mu_pred + self.C1) / (mu_gt.square() + mu_pred.square() + self.C1)
                    * (2 * covariance + self.C2) / (var_gt + var_pred + self.C2))
        if valid is None:
            dim = None if self.is_reduce_channel else (0, 2, 3)
            return ssim_map.mean(dim=dim)
        per_band = masked_mean(ssim_map, valid, dim=(-2, -1))
        return torch.nanmean(per_band) if self.is_reduce_channel else torch.nanmean(per_band, dim=0)


SSIM = StructuralSimilarity
