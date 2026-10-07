from torch import nn
import torch
from ._utils import get_valid_mask, masked_mean, prepare_images


class ErrorRelativeGlobalDimensionlessSynthesis(nn.Module):
    __name__ = 'ergas'

    def __init__(self, ratio=1.0 / 16.0):
        super().__init__()
        if ratio <= 0:
            raise ValueError("ratio must be positive")
        self.ratio = ratio

    def forward(self, gt, pred, mask=None):
        gt, pred = prepare_images(gt, pred)
        valid = None if mask is None else get_valid_mask(mask, gt)
        mse = masked_mean((gt - pred).square(), valid, dim=(-2, -1))
        mu_gt = masked_mean(gt, valid, dim=(-2, -1))
        relative_error = mse / mu_gt.square()
        relative_error = torch.where((mu_gt == 0) & (mse == 0), 0.0, relative_error)
        per_image = 100 * self.ratio * torch.nanmean(relative_error, dim=1).sqrt()
        return torch.nanmean(per_image)


ERGAS = ErrorRelativeGlobalDimensionlessSynthesis
