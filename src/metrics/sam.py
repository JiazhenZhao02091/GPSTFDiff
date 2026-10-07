from torch import nn
import torch
from typing_extensions import Literal
from ._utils import get_valid_mask, prepare_images


class SpectralAngleMapper(nn.Module):
    __name__ = 'SAM'

    def __init__(self, unit: Literal['degree', 'rad'] = 'rad'):
        super().__init__()
        if unit not in ('rad', 'degree'):
            raise ValueError("unit must be 'rad' or 'degree'")
        self.output_unit = unit

    def forward(self, gt, pred, mask=None):
        gt, pred = prepare_images(gt, pred)
        output_dtype = gt.dtype
        gt, pred = gt.double(), pred.double()
        valid = torch.ones_like(gt[:, 0], dtype=torch.bool)
        if mask is not None:
            valid = get_valid_mask(mask, gt).all(dim=1)
        norm_gt = torch.linalg.vector_norm(gt, dim=1)
        norm_pred = torch.linalg.vector_norm(pred, dim=1)
        valid = valid & (norm_gt > 0) & (norm_pred > 0)
        denominator = torch.where(valid, norm_gt * norm_pred, 1.0)
        cosine = (gt * pred).sum(dim=1) / denominator
        angles = torch.acos(cosine.clamp(-1, 1))[valid]
        if self.output_unit == 'degree':
            angles = torch.rad2deg(angles)
        return angles.mean().to(dtype=output_dtype)


SAM = SpectralAngleMapper
