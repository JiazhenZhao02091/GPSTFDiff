from torch import nn
import torch
import torch.nn.functional as F


class BoundaryRootMeanSquareError(nn.Module):
    __name__ = 'RMSE_BD'

    def __init__(self, radius=5):
        super().__init__()
        self.radius = radius

    def forward(self, gt, pred, mask=None):
        if mask is None:
            mse_value = ((gt - pred) ** 2).mean()
            return torch.sqrt(mse_value)

        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = (mask > 0).float()

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
            return torch.tensor(float('nan'), device=gt.device)

        mse_value = ((gt[valid] - pred[valid]) ** 2).mean()
        return torch.sqrt(mse_value)


RMSE_BD = BoundaryRootMeanSquareError
