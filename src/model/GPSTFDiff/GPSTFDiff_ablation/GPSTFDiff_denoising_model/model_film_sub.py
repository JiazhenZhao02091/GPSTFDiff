# import math
import cv2
import numpy as np
# from functools import partial
# import torch
# from torch import nn
# import torch.nn.functional as F
# from einops import rearrange, reduce

# # --------------------------
# # 基础组件 (与原版一致)
# # --------------------------
# def exists(x):
#     return x is not None

# def default(val, d):
#     return val if exists(val) else (d() if callable(d) else d)

# class WeightStandardizedConv2d(nn.Conv2d):
#     def forward(self, x):
#         eps = 1e-5 if x.dtype == torch.float32 else 1e-3
#         weight = self.weight
#         mean = reduce(weight, "o ... -> o 1 1 1", "mean")
#         var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
#         weight = (weight - mean) * (var + eps).rsqrt()
#         return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

# class SinusoidalPosEmb(nn.Module):
#     def __init__(self, dim: int):
#         super().__init__()
#         self.dim = dim
#     def forward(self, t: torch.Tensor) -> torch.Tensor:
#         device = t.device
#         half_dim = self.dim // 2
#         emb_scale = math.log(10000) / (half_dim - 1)
#         freqs = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
#         args = t.float()[:, None] * freqs[None, :]
#         return torch.cat([args.sin(), args.cos()], dim=-1)

# class ResBlock(nn.Module):
#     def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
#         super().__init__()
#         self.mlp = (
#             nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
#             if exists(time_emb_dim) else None
#         )
#         self.conv_1 = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
#         self.norm_1 = nn.GroupNorm(groups, dim_out)
#         self.act_1 = nn.SiLU()
#         self.conv_2 = nn.Conv2d(dim_out, dim_out, 3, padding=1)
#         self.norm_2 = nn.GroupNorm(groups, dim_out)
#         self.act_2 = nn.SiLU()
#         self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

#     def forward(self, x, time_emb=None):
#         scale_shift = None
#         if exists(self.mlp) and exists(time_emb):
#             h = self.mlp(time_emb)
#             h = rearrange(h, "b c -> b c 1 1")
#             scale_shift = h.chunk(2, dim=1)
#         h = self.conv_1(x)
#         h = self.norm_1(h)
#         if exists(scale_shift):
#             scale, shift = scale_shift
#             h = h * (scale + 1) + shift
#         h = self.act_1(h)
#         h = self.conv_2(h)
#         h = self.norm_2(h)
#         h = self.act_2(h)
#         return h + self.res_conv(x)

# # ============================================================
# # 通用融合模块 (用于消融实验)
# # ============================================================
# # class FusionLayer2d(nn.Module):
# #     def __init__(self, mode, cond_dim, out_dim):
# #         super().__init__()
# #         self.mode = mode
# #         assert mode in ['film', 'cat', 'add', 'sub'], f"Unsupported mode: {mode}"
        
# #         if mode == 'film':
# #             self.to_scale_shift = nn.Conv2d(cond_dim, out_dim * 2, 1)
# #         elif mode == 'cat':
# #             # 拼接后需要1x1卷积将通道压回out_dim以匹配主流维度
# #             self.proj = nn.Conv2d(out_dim + cond_dim, out_dim, 1)

# #     def forward(self, x, cond):
# #         if self.mode == 'film':
# #             ss = self.to_scale_shift(cond)
# #             scale, shift = ss.chunk(2, dim=1)
# #             return x * (1 + scale) + shift
# #         elif self.mode == 'cat':
# #             return self.proj(torch.cat([x, cond], dim=1))
# #         elif self.mode == 'add':
# #             return x + cond
# #         elif self.mode == 'sub':
# #             return x - cond

# class FusionLayer2d(nn.Module):
#     def __init__(self, mode, cond_dim, out_dim):
#         """
#         cond_dim: 来自条件流(cond_skip)的通道数
#         out_dim:  来自噪声流(noisy_stream)的通道数
#         """
#         super().__init__()
#         self.mode = mode
#         assert mode in ['film', 'cat', 'add', 'sub'], f"Unsupported mode: {mode}"
        
#         if mode == 'film':
#             # FiLM 需要将条件投影到噪声流通道数的两倍 (scale + shift)
#             self.to_scale_shift = nn.Conv2d(cond_dim, out_dim * 2, 1)
#         elif mode == 'cat':
#             # 拼接后投影回 out_dim
#             self.proj = nn.Conv2d(out_dim + cond_dim, out_dim, 1)
#         elif mode in ['add', 'sub']:
#             # 核心修复点：如果维度不一致(常发生在 Up 阶段)，将条件流 cond 对齐到噪声流 x 的通道数
#             if cond_dim != out_dim:
#                 self.align_cond = nn.Conv2d(cond_dim, out_dim, 1)
#             else:
#                 self.align_cond = nn.Identity()

#     def forward(self, x, cond):
#         if self.mode == 'film':
#             ss = self.to_scale_shift(cond)
#             scale, shift = ss.chunk(2, dim=1)
#             return x * (1 + scale) + shift
        
#         elif self.mode == 'cat':
#             return self.proj(torch.cat([x, cond], dim=1))
        
#         elif self.mode == 'add':
#             # 错误点修复：此处必须使用 self.align_cond(cond)
#             return x + self.align_cond(cond)
        
#         elif self.mode == 'sub':
#             # 错误点修复：此处必须使用 self.align_cond(cond)
#             return x - self.align_cond(cond)

# # ============================================================
# # MKIRA 组件 (与原版一致)
# # ============================================================
# def _gcd(a, b):
#     while b: a, b = b, a % b
#     return a

# def _channel_shuffle(x, groups: int):
#     b, c, h, w = x.size()
#     x = x.view(b, groups, c // groups, h, w).transpose(1, 2).contiguous()
#     return x.view(b, c, h, w)

# class ChannelAttention(nn.Module):
#     def __init__(self, c, ratio=16):
#         super().__init__()
#         hidden = max(1, c // ratio)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc1 = nn.Conv2d(c, hidden, 1, bias=False)
#         self.act = nn.SiLU()
#         self.fc2 = nn.Conv2d(hidden, c, 1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         avg = self.fc2(self.act(self.fc1(self.avg_pool(x))))
#         mx  = self.fc2(self.act(self.fc1(self.max_pool(x))))
#         return self.sigmoid(avg + mx)

# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super().__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         avg = torch.mean(x, dim=1, keepdim=True)
#         mx, _ = torch.max(x, dim=1, keepdim=True)
#         return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

# class MKIR_GN(nn.Module):
#     def __init__(self, in_c, out_c, stride=1, expansion=2, kernel_sizes=(1,3,5), add=True, gn_groups=8):
#         super().__init__()
#         ex_c = int(in_c * expansion)
#         self.pw1 = nn.Sequential(nn.Conv2d(in_c, ex_c, 1, bias=False), 
#                                  nn.GroupNorm(min(gn_groups, ex_c), ex_c), nn.SiLU())
#         self.branches = nn.ModuleList([
#             nn.Sequential(nn.Conv2d(ex_c, ex_c, k, stride=stride, padding=k//2, groups=ex_c, bias=False),
#                           nn.GroupNorm(min(gn_groups, ex_c), ex_c), nn.SiLU()) for k in kernel_sizes
#         ])
#         self.pw2 = nn.Sequential(nn.Conv2d(ex_c, out_c, 1, bias=False), nn.GroupNorm(min(gn_groups, out_c), out_c))
#         self.skip = nn.Identity() if in_c == out_c else nn.Conv2d(in_c, out_c, 1, bias=False)
#     def forward(self, x):
#         y = self.pw1(x)
#         z = sum([br(y) for br in self.branches])
#         z = _channel_shuffle(z, max(1, _gcd(z.shape[1], x.shape[1])))
#         return self.skip(x) + self.pw2(z)

# class MKIRA_GN(nn.Module):
#     def __init__(self, c, kernel_sizes=(1,3,5), gn_groups=8):
#         super().__init__()
#         self.ca = ChannelAttention(c)
#         self.sa = SpatialAttention(kernel_size=7)
#         self.mkir = MKIR_GN(c, c, kernel_sizes=kernel_sizes, gn_groups=gn_groups)
#     def forward(self, x):
#         x = self.ca(x) * x
#         x = self.sa(x) * x
#         return self.mkir(x)

# class ResidualGated(nn.Module):
#     def __init__(self, fn: nn.Module, init_alpha: float = 0.0):
#         super().__init__()
#         self.fn = fn
#         self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
#     def forward(self, x):
#         return x + self.alpha * self.fn(x)

# # ============================================================
# # 主模型 (支持消融模式切换)
# # ============================================================
# class PredNoiseNetMKIRA(nn.Module):
#     def __init__(
#         self,
#         dim: int = 64,
#         dim_mults=(1, 2, 4, 8),
#         channels: int = 6,
#         fusion_mode: str = 'film', # 消融实验核心参数: 'film', 'cat', 'add', 'sub'
#         use_mkira: bool = True,
#         mkira_up_indices=(1, 2),
#     ):
#         super().__init__()
#         self.fusion_mode = fusion_mode
#         time_dim = dim * 4
#         self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))

#         self.noisy_init_conv = nn.Conv2d(channels, dim, 3, padding=1)
#         self.cond_init_conv = nn.Conv2d(channels * 4, dim, 3, padding=1)

#         dims = [dim, *map(lambda m: dim * m, dim_mults)]
#         in_out = list(zip(dims[:-1], dims[1:]))

#         self.noisy_downs = nn.ModuleList([])
#         self.cond_downs = nn.ModuleList([])
#         self.down_fusions = nn.ModuleList([])

#         for ind, (dim_in, dim_out_) in enumerate(in_out):
#             is_last = ind >= (len(in_out) - 1)
#             self.noisy_downs.append(nn.ModuleList([
#                 ResBlock(dim_in, dim_in, time_emb_dim=time_dim),
#                 Downsample(dim_in, dim_out_) if not is_last else nn.Conv2d(dim_in, dim_out_, 3, padding=1),
#             ]))
#             self.cond_downs.append(nn.ModuleList([
#                 ResBlock(dim_in, dim_in),
#                 Downsample(dim_in, dim_out_) if not is_last else nn.Conv2d(dim_in, dim_out_, 3, padding=1),
#             ]))
#             # 每一层加入消融融合层
#             self.down_fusions.append(FusionLayer2d(mode=fusion_mode, cond_dim=dim_in, out_dim=dim_in))

#         mid_dim = dims[-1]
#         self.noisy_mid_block = ResBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
#         self.cond_mid_block  = ResBlock(mid_dim, mid_dim)
#         self.mid_fusion = FusionLayer2d(mode=fusion_mode, cond_dim=mid_dim, out_dim=mid_dim)

#         self.noisy_ups = nn.ModuleList([])
#         self.up_fusions  = nn.ModuleList([])
#         self.mkira_refiners = nn.ModuleList([])

#         for up_idx, (dim_in, dim_out_) in enumerate(reversed(in_out)):
#             is_last = up_idx == (len(in_out) - 1)
#             self.noisy_ups.append(nn.ModuleList([
#                 ResBlock(dim_out_ + dim_in, dim_out_, time_emb_dim=time_dim),
#                 Upsample(dim_out_, dim_in) if not is_last else nn.Conv2d(dim_out_, dim_in, 3, padding=1),
#             ]))
#             self.up_fusions.append(FusionLayer2d(mode=fusion_mode, cond_dim=dim_in, out_dim=dim_out_))
#             self.mkira_refiners.append(
#                 ResidualGated(MKIRA_GN(dim_out_)) if use_mkira and up_idx in set(mkira_up_indices) else nn.Identity()
#             )

#         self.final_conv = nn.Conv2d(dim, channels, 1)

#     def forward(self, coarse_img_01, coarse_img_02, fine_img_01, noisy_fine_img_02, time):
#         dc = coarse_img_02 - coarse_img_01
#         cond_in = torch.cat([coarse_img_01, coarse_img_02, dc, fine_img_01], dim=1)
        
#         x = self.noisy_init_conv(noisy_fine_img_02)
#         c = self.cond_init_conv(cond_in)
#         t = self.time_mlp(time)

#         noisy_skips, cond_skips = [], []
#         for (noisy_res, noisy_down), (cond_res, cond_down), fusion in zip(self.noisy_downs, self.cond_downs, self.down_fusions):
#             x = noisy_res(x, t)
#             c = cond_res(c)
#             x = fusion(x, c) # 融合应用
#             noisy_skips.append(x); cond_skips.append(c)
#             x = noisy_down(x); c = cond_down(c)

#         x = self.mid_fusion(self.noisy_mid_block(x, t), self.cond_mid_block(c))

#         for up_res, upsample in self.noisy_ups:
#             x = torch.cat([x, noisy_skips.pop()], dim=1)
#             x = up_res(x, t)
#             x = self.up_fusions[len(self.noisy_ups)-len(noisy_skips)-1](x, cond_skips.pop())
#             x = self.mkira_refiners[len(self.noisy_ups)-len(noisy_skips)-1](x)
#             x = upsample(x)

#         return self.final_conv(x)

# # --------------------------
# # 测试用例
# # --------------------------
# if __name__ == "__main__":
#     device = "cuda"
#     for mode in ['film', 'cat', 'add', 'sub']:
#         net = PredNoiseNetMKIRA(fusion_mode=mode).to(device)
#         img = torch.randn(1, 6, 128, 128).to(device)
#         t = torch.randint(0, 1000, (1,)).to(device)
#         out = net(img, img, img, img, t)
#         print(f"Mode: {mode:4s} | Output shape: {out.shape}")
#         # # 实验 1: 你的核心创新 (FiLM)
#         # model_film = PredNoiseNetMKIRA(fusion_mode='film', use_mkira=True)

#         # # 实验 2: 简单叠加 (Add)
#         # model_add = PredNoiseNetMKIRA(fusion_mode='add', use_mkira=True)

#         # # 实验 3: 简单拼接 (Concatenation)
#         # model_cat = PredNoiseNetMKIRA(fusion_mode='cat', use_mkira=True)

#         # # 实验 4: 模拟 stfdiff 的差分模式 (Subtraction)
#         # model_sub = PredNoiseNetMKIRA(fusion_mode='sub', use_mkira=True)

import math
from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, reduce

# --------------------------
# 基础组件
# --------------------------
def exists(x):
    return x is not None

def default(val, d):
    return val if exists(val) else (d() if callable(d) else d)

def cv2_resize(tensor, size=None, scale_factor=None, mode="bilinear"):
    h, w = tensor.shape[-2:]
    if size is None:
        if isinstance(scale_factor, (tuple, list)):
            scale_h, scale_w = scale_factor
        else:
            scale_h = scale_w = scale_factor
        out_h, out_w = int(h * float(scale_h)), int(w * float(scale_w))
    else:
        out_h, out_w = size
        out_h, out_w = int(out_h), int(out_w)
    interpolation = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
    }[mode]
    device, dtype = tensor.device, tensor.dtype
    array = tensor.detach().contiguous().cpu().float().numpy()
    b, c, in_h, in_w = array.shape
    array = array.reshape(b * c, in_h, in_w)
    resized = np.stack([cv2.resize(img, (out_w, out_h), interpolation=interpolation) for img in array], axis=0)
    resized = resized.reshape(b, c, out_h, out_w)
    return torch.from_numpy(resized).to(device=device, dtype=dtype)


class CV2Resize(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode="bilinear"):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return cv2_resize(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode)


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        CV2Resize(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )

def Downsample(dim, dim_out=None):
    return nn.Sequential(
        CV2Resize(scale_factor=0.5, mode="bilinear"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )

class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim) else None
        )
        self.conv_1 = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm_1 = nn.GroupNorm(groups, dim_out)
        self.act_1 = nn.SiLU()
        self.conv_2 = nn.Conv2d(dim_out, dim_out, 3, padding=1)
        self.norm_2 = nn.GroupNorm(groups, dim_out)
        self.act_2 = nn.SiLU()
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            h = self.mlp(time_emb)
            h = rearrange(h, "b c -> b c 1 1")
            scale_shift = h.chunk(2, dim=1)
        h = self.conv_1(x)
        h = self.norm_1(h)
        if exists(scale_shift):
            scale, shift = scale_shift
            h = h * (scale + 1) + shift
        h = self.act_1(h)
        h = self.conv_2(h)
        h = self.norm_2(h)
        h = self.act_2(h)
        return h + self.res_conv(x)

# ============================================================
# 通用融合模块 (彻底修复维度匹配)
# ============================================================
class FusionLayer2d(nn.Module):
    def __init__(self, mode, cond_dim, out_dim):
        super().__init__()
        self.mode = mode
        assert mode in ['film', 'cat', 'add', 'sub', 'sub_film']
        
        if mode == 'film':
            self.to_scale_shift = nn.Conv2d(cond_dim, out_dim * 2, 1)
        elif mode == 'cat':
            self.proj = nn.Conv2d(out_dim + cond_dim, out_dim, 1)
        elif mode in ['add', 'sub']:
            self.align = nn.Conv2d(cond_dim, out_dim, 1) if cond_dim != out_dim else nn.Identity()
        
        # --- 叠用模式 ---
        elif mode == 'sub_film':
            # 需要 align 层来做 sub
            self.align = nn.Conv2d(cond_dim, out_dim, 1) if cond_dim != out_dim else nn.Identity()
            # 需要 scale_shift 层来做 film
            self.to_scale_shift = nn.Conv2d(cond_dim, out_dim * 2, 1)

    def forward(self, x, cond):
        if self.mode == 'film':
            ss = self.to_scale_shift(cond)
            scale, shift = ss.chunk(2, dim=1)
            return x * (1 + scale) + shift
        
        elif self.mode == 'cat':
            return self.proj(torch.cat([x, cond], dim=1))
        
        elif self.mode == 'add':
            return x + self.align(cond)
        
        elif self.mode == 'sub':
            return x - self.align(cond)
        
        # --- 叠用逻辑：物理引导 + 动态调制 ---
        elif self.mode == 'sub_film':
            # 1. 物理残差引导
            x_res = x - self.align(cond)
            # 2. 动态特征调制
            ss = self.to_scale_shift(cond)
            scale, shift = ss.chunk(2, dim=1)
            return x_res * (1 + scale) + shift

# ============================================================
# MKIRA 组件 (保持原版逻辑)
# ============================================================
def _gcd(a, b):
    while b: a, b = b, a % b
    return a

def _channel_shuffle(x, groups: int):
    b, c, h, w = x.size()
    x = x.view(b, groups, c // groups, h, w).transpose(1, 2).contiguous()
    return x.view(b, c, h, w)

class ChannelAttention(nn.Module):
    def __init__(self, c, ratio=16):
        super().__init__()
        hidden = max(1, c // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(c, hidden, 1, bias=False)
        self.act = nn.SiLU()
        self.fc2 = nn.Conv2d(hidden, c, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = self.fc2(self.act(self.fc1(self.avg_pool(x))))
        mx  = self.fc2(self.act(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg + mx)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

class MKIR_GN(nn.Module):
    def __init__(self, in_c, out_c, stride=1, expansion=2, kernel_sizes=(1,3,5), add=True, gn_groups=8):
        super().__init__()
        ex_c = int(in_c * expansion)
        self.pw1 = nn.Sequential(nn.Conv2d(in_c, ex_c, 1, bias=False), 
                                 nn.GroupNorm(min(gn_groups, ex_c), ex_c), nn.SiLU())
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ex_c, ex_c, k, stride=stride, padding=k//2, groups=ex_c, bias=False),
                          nn.GroupNorm(min(gn_groups, ex_c), ex_c), nn.SiLU()) for k in kernel_sizes
        ])
        self.pw2 = nn.Sequential(nn.Conv2d(ex_c, out_c, 1, bias=False), nn.GroupNorm(min(gn_groups, out_c), out_c))
        self.skip = nn.Identity() if in_c == out_c else nn.Conv2d(in_c, out_c, 1, bias=False)
    def forward(self, x):
        y = self.pw1(x)
        z = sum([br(y) for br in self.branches])
        z = _channel_shuffle(z, max(1, _gcd(z.shape[1], x.shape[1])))
        return self.skip(x) + self.pw2(z)

class MKIRA_GN(nn.Module):
    def __init__(self, c, kernel_sizes=(1,3,5), gn_groups=8):
        super().__init__()
        self.ca = ChannelAttention(c)
        self.sa = SpatialAttention(kernel_size=7)
        self.mkir = MKIR_GN(c, c, kernel_sizes=kernel_sizes, gn_groups=gn_groups)
    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return self.mkir(x)

class ResidualGated(nn.Module):
    def __init__(self, fn: nn.Module, init_alpha: float = 0.0):
        super().__init__()
        self.fn = fn
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
    def forward(self, x):
        return x + self.alpha * self.fn(x)

# ============================================================
# 消融模型主类
# ============================================================
class PredNoiseNetMKIRA(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        dim_mults=(1, 2, 4, 8),
        channels: int = 6,
        fusion_mode: str = 'film',
        init_dim: int = None,
        out_dim: int = None,
        self_condition: bool = False,
        resnet_block_groups: int = 8,
        learned_variance: bool = False,
        learned_sinusoidal_cond: bool = False,
        learned_sinusoidal_dim: int = 16,
        include_f1_in_cond: bool = True,

        # ===== MKIRA options =====
        use_mkira: bool = True,
        mkira_kernel_sizes=(1, 3, 5),
        mkira_gn_groups: int = 8,
        mkira_init_alpha: float = 0.0,
        # apply MKIRA on decoder stages by index (0 = deepest up stage)
        # For dim_mults=(1,2,4) => 3 up stages: indices {0,1,2}
        mkira_up_indices=(1, 2),
    ):
        super().__init__()

        self.learned_sinusoidal_cond = False
        assert not learned_sinusoidal_cond
        self.channels = channels
        self.self_condition = self_condition   

        self.fusion_mode = fusion_mode
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))

        self.noisy_init_conv = nn.Conv2d(channels, dim, 3, padding=1)
        self.cond_init_conv = nn.Conv2d(channels * 4, dim, 3, padding=1)

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Down 阶段
        self.noisy_downs = nn.ModuleList([])
        self.cond_downs = nn.ModuleList([])
        self.down_fusions = nn.ModuleList([])

        for ind, (dim_in, dim_out_) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.noisy_downs.append(nn.ModuleList([
                ResBlock(dim_in, dim_in, time_emb_dim=time_dim),
                Downsample(dim_in, dim_out_) if not is_last else nn.Conv2d(dim_in, dim_out_, 3, padding=1),
            ]))
            self.cond_downs.append(nn.ModuleList([
                ResBlock(dim_in, dim_in),
                Downsample(dim_in, dim_out_) if not is_last else nn.Conv2d(dim_in, dim_out_, 3, padding=1),
            ]))
            self.down_fusions.append(FusionLayer2d(mode=fusion_mode, cond_dim=dim_in, out_dim=dim_in))

        # Mid 阶段
        mid_dim = dims[-1]
        self.noisy_mid_block = ResBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.cond_mid_block  = ResBlock(mid_dim, mid_dim)
        self.mid_fusion = FusionLayer2d(mode=fusion_mode, cond_dim=mid_dim, out_dim=mid_dim)

        # Up 阶段
        self.noisy_ups = nn.ModuleList([])
        self.up_fusions  = nn.ModuleList([])
        self.mkira_refiners = nn.ModuleList([])

        for up_idx, (dim_in, dim_out_) in enumerate(reversed(in_out)):
            is_last = up_idx == (len(in_out) - 1)
            self.noisy_ups.append(nn.ModuleList([
                ResBlock(dim_out_ + dim_in, dim_out_, time_emb_dim=time_dim),
                Upsample(dim_out_, dim_in) if not is_last else nn.Conv2d(dim_out_, dim_in, 3, padding=1),
            ]))
            # 这里的顺序非常重要：融合层必须对应正确的通道数
            self.up_fusions.append(FusionLayer2d(mode=fusion_mode, cond_dim=dim_in, out_dim=dim_out_))
            
            # 使用 mkira_init_alpha 初始化
            self.mkira_refiners.append(
                ResidualGated(MKIRA_GN(dim_out_), init_alpha=mkira_init_alpha)
                if use_mkira and up_idx in set(mkira_up_indices) else nn.Identity()
            )

        self.final_conv = nn.Conv2d(dim, channels, 1)

    def forward(self, coarse_img_01, coarse_img_02, fine_img_01, noisy_fine_img_02, time, x_self_cond=None,):
        # self-conditioning
        if self.self_condition:
            x_sc = default(x_self_cond, lambda: torch.zeros_like(noisy_fine_img_02))
            noisy_in = torch.cat([noisy_fine_img_02, x_sc], dim=1)
        else:
            noisy_in = noisy_fine_img_02


        
        dc = coarse_img_02 - coarse_img_01
        cond_in = torch.cat([coarse_img_01, coarse_img_02, dc, fine_img_01], dim=1)
        
        x = self.noisy_init_conv(noisy_fine_img_02)
        c = self.cond_init_conv(cond_in)
        t = self.time_mlp(time)

        noisy_skips, cond_skips = [], []
        # Down + Fusion
        for noisy_blk, cond_blk, fusion in zip(self.noisy_downs, self.cond_downs, self.down_fusions):
            n_res, n_down = noisy_blk
            c_res, c_down = cond_blk
            x = n_res(x, t)
            c = c_res(c)
            x = fusion(x, c)
            noisy_skips.append(x)
            cond_skips.append(c)
            x = n_down(x)
            c = c_down(c)

        # Mid
        x = self.noisy_mid_block(x, t)
        c = self.cond_mid_block(c)
        x = self.mid_fusion(x, c)

        # Up + Fusion (核心修正逻辑：zip保证顺序对齐)
        for up_blk, fusion, refiner in zip(self.noisy_ups, self.up_fusions, self.mkira_refiners):
            up_res, upsample = up_blk
            x = torch.cat([x, noisy_skips.pop()], dim=1)
            x = up_res(x, t)
            
            # 这里的 cond_skips 会反向弹出，与 up_fusions 的初始化顺序完美对应
            x = fusion(x, cond_skips.pop())
            x = refiner(x)
            x = upsample(x)

        return self.final_conv(x)

# --------------------------
# 测试脚本
# --------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    for mode in ['film', 'cat', 'add', 'sub', 'sub_film']:
        # 在这里测试 init_alpha
        net = PredNoiseNetMKIRA(fusion_mode=mode, mkira_init_alpha=0.1).to(device)
        img = torch.randn(1, 6, 128, 128).to(device)
        t = torch.randint(0, 1000, (1,)).to(device)
        out = net(img, img, img, img, t)
        print(f"Mode: {mode:4s} | Output shape: {out.shape} | Alpha tested.")