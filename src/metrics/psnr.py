from torch import nn
import torch
from ._utils import get_valid_mask, masked_mean, prepare_images


class PeakSignalNoiseRatio(nn.Module):
    __name__ = 'PSNR'

    def __init__(self, max_value=255):
        super().__init__()
        if max_value <= 0:
            raise ValueError("max_value must be positive")
        self.max_value = max_value

    def forward(self, gt, pred, mask=None):
        gt, pred = prepare_images(gt, pred)
        valid = None if mask is None else get_valid_mask(mask, gt)
        mse = masked_mean((gt - pred).square(), valid)
        return 20.0 * torch.log10(self.max_value / mse.sqrt())


PSNR = PeakSignalNoiseRatio
