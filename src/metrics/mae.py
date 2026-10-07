from torch import nn
from ._utils import get_valid_mask, masked_mean, prepare_images


class MeanAbsoluteError(nn.Module):
    __name__ = 'MAE'

    def __init__(self, is_reduce_channel=True):
        super().__init__()
        self.is_reduce_channel = is_reduce_channel

    def forward(self, gt, pred, mask=None):
        gt, pred = prepare_images(gt, pred)
        valid = None if mask is None else get_valid_mask(mask, gt)
        dim = None if self.is_reduce_channel else (0, 2, 3)
        return masked_mean((gt - pred).abs(), valid, dim=dim)


MAE = MeanAbsoluteError
