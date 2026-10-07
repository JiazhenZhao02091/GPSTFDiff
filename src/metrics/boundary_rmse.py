from torch import nn
import torch
import torch.nn.functional as F
from ._utils import get_valid_mask, nan_scalar, prepare_images
from .rmse import RMSE


class BoundaryRootMeanSquareError(nn.Module):
    __name__ = 'RMSE_BD'

    def __init__(self, radius=5):
        super().__init__()
        if not isinstance(radius, int) or radius < 0:
            raise ValueError("radius must be a nonnegative integer")
        self.radius = radius

    def forward(self, gt, pred, mask=None):
        gt, pred = prepare_images(gt, pred)
        if mask is None:
            return RMSE()(gt, pred)

        mask = get_valid_mask(mask, gt).to(dtype=gt.dtype)
        if not mask.any():
            return nan_scalar(gt)

        invalid = 1.0 - mask
        kernel_size = 2 * self.radius + 1
        dilated_invalid = F.max_pool2d(
            invalid,
            kernel_size=kernel_size,
            stride=1,
            padding=self.radius,
        )
        boundary = (dilated_invalid > 0).float() * mask
        if boundary.shape[1] == 1 and gt.shape[1] != 1:
            boundary = boundary.expand(-1, gt.shape[1], -1, -1)

        valid = boundary > 0
        if valid.sum() == 0:
            return nan_scalar(gt)

        mse_value = ((gt[valid] - pred[valid]) ** 2).mean()
        return torch.sqrt(mse_value)


RMSE_BD = BoundaryRootMeanSquareError
