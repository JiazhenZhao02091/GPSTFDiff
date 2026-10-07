import torch
import torch.nn as nn
from .pred_resnet import PredNoiseNet
from .simple_UNet import SimpleUNet


def _get_denoiser(simple = None):
    if simple:
        return SimpleUNet(
            channels=6,
            dim=64,
            dim_mults=(1, 2, 4, 8),
            resnet_block_groups=8,
            learned_variance=True,
            self_condition=False,
        )
    else:
        return PredNoiseNet(
            channels=6,
            dim=64,
            dim_mults=(1, 2, 4, 8),
            resnet_block_groups=8,
            learned_variance=True,
            self_condition=False,
        )

class LaDMSTF(nn.Module):
    def __init__(self):
        super().__init__()
        self.denoiser1 = _get_denoiser(simple=True)
        self.denoiser2 = _get_denoiser(simple=False)
        self.denoiser3 = _get_denoiser(simple=False)
        pass
    def sample(self, coarse_img_01, coarse_img_02, fine_img_01, noisy_fine_img_02, time):
        pass

class Denoiser(nn.Module):
    def __init__(self):
        pass
