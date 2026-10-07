from torch import nn
import torch
from ._utils import band_statistics, prepare_images


class UniversalImageQualityIndex(nn.Module):
    __name__ = 'UIQI'

    def __init__(self, is_reduce_channel=True):
        super().__init__()
        self.is_reduce_channel = is_reduce_channel

    def forward(self, gt, pred, mask=None):
        gt, pred = prepare_images(gt, pred)
        mu_gt, mu_pred, covariance, var_gt, var_pred = band_statistics(gt, pred, mask)
        variance_sum = var_gt + var_pred
        mean_square_sum = mu_gt.square() + mu_pred.square()
        contrast = torch.where(variance_sum == 0, 1.0, 2 * covariance / variance_sum)
        luminance = torch.where(mean_square_sum == 0, 1.0, 2 * mu_gt * mu_pred / mean_square_sum)
        scores = (contrast * luminance).clamp(-1, 1)
        result = torch.nanmean(scores) if self.is_reduce_channel else torch.nanmean(scores, dim=0)
        return result.to(dtype=gt.dtype)


UIQI = UniversalImageQualityIndex
