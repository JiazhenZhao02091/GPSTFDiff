import torch
import torch.nn as nn
from .pred_resnet import PredNoiseNet # for 2
from .simple_UNet import SimpleUNet   # for 1

"""
    c1 c2 f1 --> f2
    TODO: time range
    TODO: three condition denoiser
    down(x1) --> x2   down(x2) --> x3
    第一个去噪网络,恢复出来 x3, 给定c2的前提 
    
"""
def _get_denoiser(simple = None):
    if simple: # 获得简单U-Net
        return SimpleUNet(
            channels=6,
            dim=64,
            dim_mults=(1, 2, 4, 8),
            resnet_block_groups=8,
            learned_variance=True,
            self_condition=False,
        )
    else: # 获得预测噪声网络
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
        self.denoiser1 = _get_denoiser(simple=True)  # SimpleNet
        self.denoiser2 = _get_denoiser(simple=False) # PredNoiseNet
        self.denoiser3 = _get_denoiser(simple=False) # PredNoiseNet
        pass
    def sample(self, coarse_img_01, coarse_img_02, fine_img_01, noisy_fine_img_02, time):
        pass
"""
    base U-Net
"""
class Denoiser(nn.Module):
    def __init__(self):
        pass
