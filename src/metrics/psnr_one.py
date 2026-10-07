from torch import nn
import torch
from ._utils import get_valid_mask, masked_mean, prepare_images


class PeakSignalNoiseRatio(nn.Module):
    __name__ = 'PSNRONE'

    def __init__(self, max_value=255, is_reduce_channel=True):
        super().__init__()
        if max_value <= 0:
            raise ValueError("max_value must be positive")
        self.max_value = max_value
        self.is_reduce_channel = is_reduce_channel

    def forward(self, gt, pred, mask=None):
        gt, pred = prepare_images(gt, pred)
        valid = None if mask is None else get_valid_mask(mask, gt)
        dim = None if self.is_reduce_channel else (0, 2, 3)
        mse = masked_mean((gt - pred).square(), valid, dim=dim)
        return 20.0 * torch.log10(self.max_value / mse.sqrt())


PSNRONE = PeakSignalNoiseRatio
