import torch


def prepare_images(gt, pred):
    if gt.ndim != 4 or gt.shape != pred.shape:
        raise ValueError("gt and pred must have the same [B, C, H, W] shape")
    if gt.device != pred.device:
        raise ValueError("gt and pred must be on the same device")
    dtype = torch.promote_types(gt.dtype, pred.dtype)
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    return gt.to(dtype=dtype), pred.to(dtype=dtype)


def get_valid_mask(mask, reference):
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dim() != 4:
        raise ValueError("mask must have shape [B, H, W] or [B, C, H, W]")
    if mask.shape[0] != reference.shape[0] or mask.shape[-2:] != reference.shape[-2:]:
        raise ValueError("mask batch and spatial dimensions must match the images")
    if mask.shape[1] == 1:
        mask = mask.expand(-1, reference.shape[1], -1, -1)
    elif mask.shape[1] != reference.shape[1]:
        raise ValueError("mask channels must be 1 or match the image channels")
    return mask.to(device=reference.device) > 0


def masked_mean(values, valid_mask=None, dim=None, keepdim=False):
    if valid_mask is None:
        return values.mean(dim=dim, keepdim=keepdim)
    values = torch.where(valid_mask, values, 0)
    count = valid_mask.sum(dim=dim, keepdim=keepdim)
    result = values.sum(dim=dim, keepdim=keepdim) / count.clamp(min=1)
    return torch.where(count > 0, result, torch.full_like(result, float("nan")))


def band_statistics(gt, pred, mask=None):
    gt, pred = prepare_images(gt, pred)
    gt, pred = gt.double(), pred.double()
    valid = None if mask is None else get_valid_mask(mask, gt)
    mu_gt = masked_mean(gt, valid, dim=(-2, -1), keepdim=True)
    mu_pred = masked_mean(pred, valid, dim=(-2, -1), keepdim=True)
    centered_gt = gt - mu_gt
    centered_pred = pred - mu_pred
    covariance = masked_mean(centered_gt * centered_pred, valid, dim=(-2, -1))
    var_gt = masked_mean(centered_gt.square(), valid, dim=(-2, -1))
    var_pred = masked_mean(centered_pred.square(), valid, dim=(-2, -1))
    return mu_gt[..., 0, 0], mu_pred[..., 0, 0], covariance, var_gt, var_pred


def nan_scalar(reference):
    return torch.full((), float("nan"), device=reference.device, dtype=reference.dtype)
