from torch import nn
import torch
from ._utils import band_statistics, prepare_images


class CorrelationCoefficient(nn.Module):
    __name__ = 'CC'

    def __init__(self, is_reduce_channel=True):
        super().__init__()
        self.is_reduce_channel = is_reduce_channel

    def forward(self, gt, pred, mask=None):
        gt, pred = prepare_images(gt, pred)
        _, _, covariance, var_gt, var_pred = band_statistics(gt, pred, mask)
        denominator = (var_gt * var_pred).sqrt()
        scores = covariance / denominator
        scores = torch.where(denominator > 0, scores.clamp(-1, 1), float("nan"))
        result = torch.nanmean(scores) if self.is_reduce_channel else torch.nanmean(scores, dim=0)
        return result.to(dtype=gt.dtype)


CC = CorrelationCoefficient
