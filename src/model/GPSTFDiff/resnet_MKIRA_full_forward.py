import math
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, reduce

"""
    forward参数为 [coarse_img_01, coarse_img_02, fine_img_01, noisy_fine_img_02, t, x_self_cond]
"""

# --------------------------
# helpers
# --------------------------
def exists(x):
    return x is not None

def default(val, d):
    return val if exists(val) else (d() if callable(d) else d)

def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )

def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


# --------------------------
# weight standardized conv
# --------------------------
class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


# --------------------------
# time embedding
# --------------------------
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


# --------------------------
# blocks
# --------------------------
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


class FiLM2d(nn.Module):
    def __init__(self, cond_dim, out_dim):
        super().__init__()
        self.to_scale_shift = nn.Conv2d(cond_dim, out_dim * 2, 1)

    def forward(self, x, cond):
        ss = self.to_scale_shift(cond)
        scale, shift = ss.chunk(2, dim=1)
        return x * (1 + scale) + shift


# ============================================================
# MKIRA (GN版) + 门控残差：用于 decoder 细节精炼，扩散更稳
# ============================================================
def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _channel_shuffle(x, groups: int):
    b, c, h, w = x.size()
    assert c % groups == 0
    x = x.view(b, groups, c // groups, h, w).transpose(1, 2).contiguous()
    return x.view(b, c, h, w)

class ChannelAttention(nn.Module):
    def __init__(self, c, ratio=16):
        super().__init__()
        ratio = min(ratio, c)
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
        assert kernel_size in (3, 7, 11)
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg, mx], dim=1)
        return self.sigmoid(self.conv(a))

class MultiKernelDepthwiseConvGN(nn.Module):
    def __init__(self, c, kernel_sizes=(1, 3, 5), stride=1, gn_groups=8):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, k, stride=stride, padding=k//2, groups=c, bias=False),
                nn.GroupNorm(num_groups=min(gn_groups, c), num_channels=c),
                nn.SiLU(),
            )
            for k in kernel_sizes
        ])

    def forward(self, x):
        return [br(x) for br in self.branches]

class MKIR_GN(nn.Module):
    """
    MKIR: PWC expand -> MKDC (multi-kernel DW) -> PWC project + channel shuffle + residual
    """
    def __init__(self, in_c, out_c, stride=1, expansion=2, kernel_sizes=(1,3,5), add=True, gn_groups=8):
        super().__init__()
        assert stride in (1,2)
        self.stride = stride
        self.in_c = in_c
        self.out_c = out_c
        self.add = add

        ex_c = int(in_c * expansion)

        self.pw1 = nn.Sequential(
            nn.Conv2d(in_c, ex_c, 1, bias=False),
            nn.GroupNorm(num_groups=min(gn_groups, ex_c), num_channels=ex_c),
            nn.SiLU(),
        )

        self.mkdc = MultiKernelDepthwiseConvGN(ex_c, kernel_sizes=kernel_sizes, stride=stride, gn_groups=gn_groups)
        combined = ex_c if add else ex_c * len(kernel_sizes)

        self.pw2 = nn.Sequential(
            nn.Conv2d(combined, out_c, 1, bias=False),
            nn.GroupNorm(num_groups=min(gn_groups, out_c), num_channels=out_c),
        )

        self.use_skip = (stride == 1)
        self.skip = nn.Identity() if (in_c == out_c) else nn.Conv2d(in_c, out_c, 1, bias=False)

    def forward(self, x):
        y = self.pw1(x)
        outs = self.mkdc(y)
        if self.add:
            z = 0
            for o in outs:
                z = z + o
        else:
            z = torch.cat(outs, dim=1)

        g = max(1, _gcd(z.shape[1], self.out_c))
        z = _channel_shuffle(z, g)
        z = self.pw2(z)

        if self.use_skip:
            return self.skip(x) + z
        return z

class MKIRA_GN(nn.Module):
    """
    MKIRA(x) = MKIR(SA(CA(x)))
    """
    def __init__(self, c, kernel_sizes=(1,3,5), gn_groups=8):
        super().__init__()
        self.ca = ChannelAttention(c)
        self.sa = SpatialAttention(kernel_size=7)
        self.mkir = MKIR_GN(c, c, stride=1, expansion=2, kernel_sizes=kernel_sizes, add=True, gn_groups=gn_groups)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return self.mkir(x)

class ResidualGated(nn.Module):
    """
    y = x + alpha * f(x), alpha init to 0 => starts as exact baseline
    """
    def __init__(self, fn: nn.Module, init_alpha: float = 0.0):
        super().__init__()
        self.fn = fn
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        return x + self.alpha * self.fn(x)


# --------------------------
# main model (FiLM baseline + optional MKIRA refinement)
# --------------------------
class PredNoiseNetMKIRA(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        init_dim: int = None,
        out_dim: int = None,
        dim_mults=(1, 2, 4, 8),
        channels: int = 6,
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

        self.channels = channels
        self.self_condition = self_condition
        self.learned_sinusoidal_cond = False
        assert not learned_sinusoidal_cond

        # time embedding
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        input_channels = channels * (2 if self_condition else 1)
        init_dim = default(init_dim, dim)

        # noisy stream input
        self.noisy_init_conv = nn.Conv2d(input_channels, init_dim, 3, padding=1)

        # conditional stream input: c1, c2, dc, (optional f1)
        cond_in_ch = channels * 3 + (channels if include_f1_in_cond else 0)
        self.include_f1_in_cond = include_f1_in_cond
        self.cond_init_conv = nn.Conv2d(cond_in_ch, init_dim, 3, padding=1)

        # multiscale dims
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        # down blocks
        self.noisy_downs = nn.ModuleList([])
        self.cond_downs = nn.ModuleList([])
        self.down_films = nn.ModuleList([])

        for ind, (dim_in, dim_out_) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.noisy_downs.append(nn.ModuleList([
                ResBlock(dim_in, dim_in, time_emb_dim=time_dim, groups=resnet_block_groups),
                Downsample(dim_in, dim_out_) if not is_last else nn.Conv2d(dim_in, dim_out_, 3, padding=1),
            ]))

            self.cond_downs.append(nn.ModuleList([
                ResBlock(dim_in, dim_in, time_emb_dim=None, groups=resnet_block_groups),
                Downsample(dim_in, dim_out_) if not is_last else nn.Conv2d(dim_in, dim_out_, 3, padding=1),
            ]))

            self.down_films.append(FiLM2d(cond_dim=dim_in, out_dim=dim_in))

        # mid
        mid_dim = dims[-1]
        self.noisy_mid_block = ResBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_block_groups)
        self.cond_mid_block  = ResBlock(mid_dim, mid_dim, time_emb_dim=None, groups=resnet_block_groups)
        self.mid_film = FiLM2d(cond_dim=mid_dim, out_dim=mid_dim)

        # up blocks
        self.noisy_ups = nn.ModuleList([])
        self.up_films  = nn.ModuleList([])

        # MKIRA refiners aligned with each up stage
        self.use_mkira = bool(use_mkira)
        self.mkira_refiners = nn.ModuleList([])

        for up_idx, (dim_in, dim_out_) in enumerate(reversed(in_out)):
            is_last = up_idx == (len(in_out) - 1)

            self.noisy_ups.append(nn.ModuleList([
                ResBlock(dim_out_ + dim_in, dim_out_, time_emb_dim=time_dim, groups=resnet_block_groups),
                Upsample(dim_out_, dim_in) if not is_last else nn.Conv2d(dim_out_, dim_in, 3, padding=1),
            ]))

            self.up_films.append(FiLM2d(cond_dim=dim_in, out_dim=dim_out_))

            # apply MKIRA only on selected up stages (typically high-res ones)
            if self.use_mkira and (up_idx in set(mkira_up_indices)):
                self.mkira_refiners.append(
                    ResidualGated(
                        MKIRA_GN(dim_out_, kernel_sizes=mkira_kernel_sizes, gn_groups=mkira_gn_groups),
                        init_alpha=mkira_init_alpha,
                    )
                )
            else:
                self.mkira_refiners.append(nn.Identity())

        default_out_dim = channels * (2 if learned_variance else 1)
        self.out_dim = default(out_dim, default_out_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    def forward(
        self,
        coarse_img_01,
        coarse_img_02,
        fine_img_01,
        noisy_fine_img_02,
        time,
        x_self_cond=None,
    ):
        # self-conditioning
        if self.self_condition:
            x_sc = default(x_self_cond, lambda: torch.zeros_like(noisy_fine_img_02))
            noisy_in = torch.cat([noisy_fine_img_02, x_sc], dim=1)
        else:
            noisy_in = noisy_fine_img_02

        # conditional tensor: [c1, c2, dc, (f1)]
        dc = coarse_img_02 - coarse_img_01
        if self.include_f1_in_cond:
            cond_in = torch.cat([coarse_img_01, coarse_img_02, dc, fine_img_01], dim=1)
        else:
            cond_in = torch.cat([coarse_img_01, coarse_img_02, dc], dim=1)

        x = self.noisy_init_conv(noisy_in)
        c = self.cond_init_conv(cond_in)

        t = self.time_mlp(time)

        noisy_skips = []
        cond_skips  = []

        # down + FiLM
        for (noisy_res, noisy_down), (cond_res, cond_down), film in zip(
            self.noisy_downs, self.cond_downs, self.down_films
        ):
            x = noisy_res(x, t)
            c = cond_res(c)
            x = film(x, c)

            noisy_skips.append(x)
            cond_skips.append(c)

            x = noisy_down(x)
            c = cond_down(c)

        # mid
        x = self.noisy_mid_block(x, t)
        c = self.cond_mid_block(c)
        x = self.mid_film(x, c)

        # up + FiLM (+ MKIRA refine)
        for up_idx, ((up_res, upsample), up_film, refiner) in enumerate(
            zip(self.noisy_ups, self.up_films, self.mkira_refiners)
        ):
            x_skip = noisy_skips.pop()
            c_skip = cond_skips.pop()

            x = torch.cat([x, x_skip], dim=1)
            x = up_res(x, t)
            x = up_film(x, c_skip)

            # MKIRA refine at selected decoder stages (gated residual, alpha init 0)
            x = refiner(x)

            x = upsample(x)

        return self.final_conv(x)


if __name__ == "__main__":
    net = PredNoiseNetMKIRA(
        dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4),
        use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0
    )
    c1 = torch.randn(1, 6, 256, 256)
    c2 = torch.randn(1, 6, 256, 256)
    f1 = torch.randn(1, 6, 256, 256)
    noisy_f2 = torch.randn(1, 6, 256, 256)
    time = torch.randint(0, 1000, (1,))
    out = net(c1, c2, f1, noisy_f2, time)
    print(out.shape)
