# =========================
# FSDFormer as Diffusion Backbone (AdaLN + t embedding)
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn import init as init


"""
Diffusion TODO done:
- forward(noisy_img, t, condition)  condition = (coarse1, coarse2, fine_1)
- timestep embedding + AdaLayerNorm injection (DiT/U-ViT style)
- remove Tanh at output head (predict eps/residual)
"""


def window_partition(x, window_size):
    """spilt 4D tensor into windows"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """reverse windows back to 4D tensor"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# -------------------------
# Diffusion: timestep embedding (sin/cos) + MLP
# -------------------------
def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    """
    timesteps: (B,) int/float tensor
    return: (B, dim)
    """
    if timesteps.dim() != 1:
        timesteps = timesteps.view(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimeEmbed(nn.Module):
    """t -> (B, time_dim) -> MLP -> (B, hidden)"""
    def __init__(self, time_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor):
        return self.mlp(t)


class AdaLayerNorm(nn.Module):
    """
    AdaLN / FiLM for diffusion transformers (DiT-like):
    y = LN(x) * (1 + gamma) + beta
    cond: (B, cond_dim)
    """
    def __init__(self, normalized_shape: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        self.mod = nn.Linear(cond_dim, 2 * normalized_shape)

        # init to near-identity (stable at start)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        # x: (B, L, C)
        h = self.norm(x)
        gamma_beta = self.mod(cond)  # (B, 2C)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        return h * (1.0 + gamma[:, None, :]) + beta[:, None, :]


# -------------------------
# Original modules (light edits for cond injection)
# -------------------------
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=256, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.patches_resolution = [self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchMerging(nn.Module):
    """ Patch Merging Layer """
    def __init__(self, input_resolution, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class frequency_selection(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.GELU):
        super(frequency_selection, self).__init__()
        self.act_fft = act_method()

        hid_dim = dim * dw
        self.complex_weight1_real = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(dim, hid_dim))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(hid_dim, dim))
        self.complex_weight2_imag = nn.Parameter(torch.Tensor(hid_dim, dim))
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(16))

        self.norm = norm

    def forward(self, x):
        _, hw, _ = x.size()
        hh = int(math.sqrt(hw))
        x1 = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)

        y = torch.fft.rfft2(x1, norm=self.norm)
        dim1 = 1
        weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
        weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)

        y = rearrange(y, 'b c h w -> b h w c')
        y = y @ weight1

        y = torch.cat([y.real, y.imag], dim=dim1)

        y = self.act_fft(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim1)
        y = torch.complex(y_real, y_imag)
        y = y @ weight2

        y = rearrange(y, 'b h w c -> b c h w')

        y = torch.fft.irfft2(y, s=(hh, hh), norm=self.norm)
        y = rearrange(y, ' b c h w -> b (h w) c', h=hh, w=hh)
        return y


class AgentAttention(nn.Module):
    """ Agent Attention 模块 """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 shift_size=0, agent_num=49, **kwargs):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.agent_num = agent_num
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)

        # Agent attention biases
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 8, 8))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 8, 8))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0], 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1]))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))

        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)

        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

    def forward(self, x, mask=None):
        b, n, c = x.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads

        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        agent_tokens = self.pool(q.reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        position_bias1 = nn.functional.interpolate(self.an_bias, size=self.window_size, mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2

        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2

        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DFAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 shift_size=0, agent_num=49, **kwargs):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.win = window_size[0] * window_size[1]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.shift_size = shift_size

        self.agent_num = agent_num
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0], 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1]))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

        self.Q = nn.Linear(dim, dim, bias=qkv_bias)
        self.K = nn.Linear(dim, dim, bias=qkv_bias)
        self.V = nn.Linear(dim, dim, bias=qkv_bias)
        self.A1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.A2 = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, x_F1, x_C2, x_C1, mask=None):
        b, n, c = x_F1.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads

        q = self.Q(x_F1)
        k = self.K(x_F1 + x_C2 - x_C1)
        v = self.V(x_C2 - x_C1)
        agent_tokens1 = self.pool(self.A1(x_C1).reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        agent_tokens2 = self.pool(self.A2(x_C2).reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)

        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens1 = agent_tokens1.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens2 = agent_tokens2.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        position_bias1 = nn.functional.interpolate(self.an_bias, size=self.window_size, mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)

        position_bias = position_bias1 + position_bias2
        agent_attn2 = self.softmax((agent_tokens2 * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn2 = self.attn_drop(agent_attn2)
        agent_v = agent_attn2 @ v

        agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)

        agent_bias = agent_bias1 + agent_bias2
        q_attn = self.softmax((q * self.scale) @ agent_tokens1.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        q = q.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.dwc(q).permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SEATBlock(nn.Module):
    """ SEAT Block (AdaLN injected) """
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, agent_num=64,
                 cond_dim: int = None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # AdaLN
        if cond_dim is None:
            # fallback: vanilla LN (but diffusion usage should pass cond_dim)
            self.norm1 = norm_layer(dim)
            self.norm2 = norm_layer(dim)
            self.use_ada = False
        else:
            self.norm1 = AdaLayerNorm(dim, cond_dim)
            self.norm2 = AdaLayerNorm(dim, cond_dim)
            self.use_ada = True

        self.attn = AgentAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            agent_num=agent_num)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.FFT = frequency_selection(dim)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))

            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def _norm(self, norm, x, cond):
        if self.use_ada:
            return norm(x, cond)
        return norm(x)

    def forward(self, x, cond=None):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self._norm(self.norm1, x, cond)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)

        h = self._norm(self.norm2, x, cond)
        x = x + self.drop_path(self.mlp(h) + self.FFT(h))
        return x


class SDFBlock(nn.Module):
    """ SDF Block (AdaLN on FFN part; main norms in SDFLayer) """
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, agent_num=64,
                 cond_dim: int = None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.attn = DFAttention(dim=dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                                agent_num=agent_num)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if cond_dim is None:
            self.norm3 = norm_layer(dim)
            self.use_ada = False
        else:
            self.norm3 = AdaLayerNorm(dim, cond_dim)
            self.use_ada = True

        self.mlp = Mlp(in_features=dim, hidden_features=dim * 4, act_layer=act_layer, drop=drop)
        self.FFT = frequency_selection(dim)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))

            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def _norm(self, x, cond):
        if self.use_ada:
            return self.norm3(x, cond)
        return self.norm3(x)

    def forward(self, x_F1, x_C2, x_C1, cond=None):
        H, W = self.input_resolution
        B, L, C = x_F1.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x_F1 + x_C2 - x_C1

        x_F1 = x_F1.view(B, H, W, C)
        x_C2 = x_C2.view(B, H, W, C)
        x_C1 = x_C1.view(B, H, W, C)

        images = []
        for x in [x_F1, x_C2, x_C1]:
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x

            x_windows = window_partition(shifted_x, self.window_size)
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
            images.append(x_windows)

        attn_windows = self.attn(images[0], images[1], images[2], mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = x + shortcut

        h = self._norm(x, cond)
        x = x + self.mlp(h) + self.FFT(h)
        return x


class SEATLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, agent_num=64,
                 cond_dim: int = None):
        super().__init__()
        self.blocks = nn.ModuleList([
            SEATBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path,
                norm_layer=norm_layer, agent_num=agent_num,
                cond_dim=cond_dim
            )
            for i in range(depth)
        ])

    def forward(self, x, cond=None):
        for blk in self.blocks:
            x = blk(x, cond=cond)
        return x


class SDFLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, agent_num=64,
                 cond_dim: int = None):
        super().__init__()
        self.blocks = nn.ModuleList([
            SDFBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path,
                norm_layer=norm_layer, agent_num=agent_num,
                cond_dim=cond_dim
            )
            for i in range(depth)
        ])

        if cond_dim is None:
            self.norm1 = norm_layer(dim)
            self.norm2 = norm_layer(dim)
            self.norm3 = norm_layer(dim)
            self.use_ada = False
        else:
            self.norm1 = AdaLayerNorm(dim, cond_dim)
            self.norm2 = AdaLayerNorm(dim, cond_dim)
            self.norm3 = AdaLayerNorm(dim, cond_dim)
            self.use_ada = True

    def _n1(self, x, cond): return self.norm1(x, cond) if self.use_ada else self.norm1(x)
    def _n2(self, x, cond): return self.norm2(x, cond) if self.use_ada else self.norm2(x)
    def _n3(self, x, cond): return self.norm3(x, cond) if self.use_ada else self.norm3(x)

    def forward(self, x_F1, x_C2, x_C1, cond=None):
        x_lr = self._n1(x_F1, cond)
        x_hr = self._n2(x_C2, cond)
        x_c  = self._n3(x_C1, cond)

        x_diff = x_lr
        for blk in self.blocks:
            x_diff = blk(x_lr, x_hr, x_c, cond=cond)
        return x_diff


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resolution, downsample, cur_depth, agent_num, cond_dim: int = None):
        super(EncoderBlock, self).__init__()
        self.layer = SEATLayer(
            dim=in_channels,
            input_resolution=(resolution, resolution),
            depth=cur_depth,
            num_heads=max(1, in_channels // 32),
            window_size=8,
            mlp_ratio=4,
            qkv_bias=True, qk_scale=None,
            drop=0., attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm, agent_num=agent_num,
            cond_dim=cond_dim
        )

        if downsample is not None:
            self.downsample = downsample((resolution, resolution), in_channels, out_channels)
        else:
            self.downsample = None

    def forward(self, x, cond=None):
        x_o = self.layer(x, cond=cond)
        if self.downsample is not None:
            x_o = self.downsample(x_o)
        return x_o


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resolution, cur_depth, agent_num, cond_dim: int = None):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 3, 1, 1),
            nn.PixelShuffle(2)
        )

        self.layer = SDFLayer(
            dim=out_channels, input_resolution=(resolution, resolution),
            depth=2, num_heads=max(1, in_channels // 32), window_size=8, mlp_ratio=4,
            qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.,
            norm_layer=nn.LayerNorm, agent_num=agent_num,
            cond_dim=cond_dim
        )

        self.layer2 = SEATLayer(
            dim=out_channels, input_resolution=(resolution, resolution),
            depth=cur_depth, num_heads=max(1, out_channels // 32), window_size=8, mlp_ratio=4,
            qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.,
            norm_layer=nn.LayerNorm, agent_num=agent_num,
            cond_dim=cond_dim
        )

        self.proj1 = nn.Linear(out_channels * 3, out_channels)

    def forward(self, x_fine0, x_coarse1, x_coarse0, x_fine1, cond=None):
        B, L, C = x_fine1.shape

        x_f1 = x_fine1.transpose(1, 2).view(B, C, self.resolution // 2, self.resolution // 2)
        x_f1 = self.up(x_f1).flatten(2).transpose(1, 2)

        x_f0 = self.layer(x_fine0, x_coarse1, x_coarse0, cond=cond)
        x = torch.cat([x_coarse0, x_f0, x_f1], dim=2)
        x = self.proj1(x)
        x = self.layer2(x, cond=cond)

        return x


class Encoder(nn.Module):
    def __init__(self, down_scale=2, in_dim=64, depths=(2, 2, 6, 2), agent_num=[9, 16, 64, 64],
                 in_channels=4, cond_dim: int = None, img_size=256):
        super(Encoder, self).__init__()
        self.inc = PatchEmbed(img_size=img_size, patch_size=down_scale, in_chans=in_channels, embed_dim=in_dim,
                              norm_layer=nn.LayerNorm)
        self.enc1 = EncoderBlock(in_channels=in_dim, out_channels=in_dim, resolution=img_size // down_scale,
                                 downsample=PatchMerging, cur_depth=depths[0], agent_num=agent_num[0], cond_dim=cond_dim)
        self.enc2 = EncoderBlock(in_channels=in_dim, out_channels=in_dim * 2, resolution=(img_size // 2) // down_scale,
                                 downsample=PatchMerging, cur_depth=depths[1], agent_num=agent_num[1], cond_dim=cond_dim)
        self.enc3 = EncoderBlock(in_channels=in_dim * 2, out_channels=in_dim * 4, resolution=(img_size // 4) // down_scale,
                                 downsample=PatchMerging, cur_depth=depths[2], agent_num=agent_num[2], cond_dim=cond_dim)
        self.enc4 = EncoderBlock(in_channels=in_dim * 4, out_channels=in_dim * 8, resolution=(img_size // 8) // down_scale,
                                 downsample=PatchMerging, cur_depth=depths[3], agent_num=agent_num[3], cond_dim=cond_dim)

    def forward(self, x, cond=None):
        x1 = self.inc(x)
        x2 = self.enc1(x1, cond=cond)
        x3 = self.enc2(x2, cond=cond)
        x4 = self.enc3(x3, cond=cond)
        x5 = self.enc4(x4, cond=cond)
        return x1, x2, x3, x4, x5


class Decoder(nn.Module):
    def __init__(self, in_dim=64, down_scale=2, depths=(2, 2, 6, 2), agent_num=[9, 16, 64, 64],
                 out_channels=3, cond_dim: int = None, img_size=256):
        super(Decoder, self).__init__()
        self.down_scale = down_scale
        self.img_size = img_size

        self.dec1 = DecoderBlock(in_dim * 8, in_dim * 4, (img_size // 8) // down_scale, depths[3], agent_num=agent_num[3], cond_dim=cond_dim)
        self.dec2 = DecoderBlock(in_dim * 4, in_dim * 2, (img_size // 4) // down_scale, depths[2], agent_num=agent_num[2], cond_dim=cond_dim)
        self.dec3 = DecoderBlock(in_dim * 2, in_dim,     (img_size // 2) // down_scale, depths[1], agent_num=agent_num[1], cond_dim=cond_dim)
        self.dec4 = DecoderBlock(in_dim,     in_dim,     (img_size // 1) // down_scale, depths[0], agent_num=agent_num[0], cond_dim=cond_dim)

        # Diffusion head: NO Tanh
        self.outc = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(in_dim, out_channels, 3, 1, 1),
        )

        self.layer = SDFLayer(dim=in_dim * 8, input_resolution=((img_size // 16) // down_scale, (img_size // 16) // down_scale),
                              depth=2, num_heads=max(1, (in_dim * 8) // 32), window_size=8, mlp_ratio=4,
                              qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.,
                              norm_layer=nn.LayerNorm, agent_num=agent_num[3],
                              cond_dim=cond_dim)

    def forward(self, fine_fea, coarse_fea1, coarse_fea0, cond=None):
        x0 = self.layer(fine_fea[4], coarse_fea1[4], coarse_fea0[4], cond=cond)

        x1 = self.dec1(fine_fea[3], coarse_fea1[3], coarse_fea0[3], x0, cond=cond)
        x2 = self.dec2(fine_fea[2], coarse_fea1[2], coarse_fea0[2], x1, cond=cond)
        x3 = self.dec3(fine_fea[1], coarse_fea1[1], coarse_fea0[1], x2, cond=cond)
        x4 = self.dec4(fine_fea[0], coarse_fea1[0], coarse_fea0[0], x3, cond=cond)

        B, L, C = x4.shape
        x4 = x4.transpose(1, 2).view(B, C, self.img_size // self.down_scale, self.img_size // self.down_scale)
        output = self.outc(x4)   # (B, out_channels, H, W)
        return output


# -------------------------
# Diffusion Backbone Wrapper
# -------------------------
class FSDFormerDiffusion(nn.Module):
    """
    x_t = noisy_fine2 (6ch)
    condition = (c1=coarse_t1, c2=coarse_t2, f1=fine_t1) all 6ch
    """
    def __init__(
        self,
        in_dim=64,
        down_scale=2,
        depths=(2, 2, 6, 2),
        agent_num=[9, 16, 64, 64],
        channels=6,          # all are 6ch
        img_size=256,
        time_cond_dim=256,
    ):
        super().__init__()
        self.img_size = img_size
        self.channels = channels

        self.time_dim = time_cond_dim
        self.t_proj_in = nn.Linear(time_cond_dim, time_cond_dim)
        self.time_mlp = TimeEmbed(time_dim=time_cond_dim, hidden_dim=time_cond_dim)
        cond_dim = time_cond_dim

        # encode noisy_fine2
        self.encoder_noisy = Encoder(
            down_scale=down_scale, in_dim=in_dim, depths=depths,
            agent_num=agent_num, in_channels=channels, cond_dim=cond_dim, img_size=img_size
        )
        # encode fine_t1 (f1)
        self.encoder_f1 = Encoder(
            down_scale=down_scale, in_dim=in_dim, depths=depths,
            agent_num=agent_num, in_channels=channels, cond_dim=cond_dim, img_size=img_size
        )
        # encode coarse_t1 / coarse_t2 (shared weights is ok; here share one encoder)
        self.encoder_coarse = Encoder(
            down_scale=down_scale, in_dim=in_dim, depths=depths,
            agent_num=agent_num, in_channels=channels, cond_dim=cond_dim, img_size=img_size
        )

        self.decoder = Decoder(
            in_dim=in_dim, down_scale=down_scale, depths=depths,
            agent_num=agent_num, out_channels=channels, cond_dim=cond_dim, img_size=img_size
        )
        
    def _parse_condition(self, condition: torch.Tensor):
        # condition can be tuple/list or a concat tensor (B, 18, H, W)
        if isinstance(condition, (tuple, list)):
            assert len(condition) == 3
            c1, c2, f1 = condition
            return c1, c2, f1

        assert torch.is_tensor(condition), "condition must be tuple/list or torch.Tensor"
        assert condition.shape[1] == self.channels * 3, \
            f"expect condition channels={self.channels*3}, got {condition.shape[1]}"
        c1, c2, f1 = torch.split(condition, self.channels, dim=1)
        return c1, c2, f1
    
    def forward(self, noisy_fine2, t, condition):
        """
        noisy_fine2: (B, 6, H, W)  -> x_t
        t: (B,)
        condition: (c1, c2, f1) each (B, 6, H, W)
        """
        c1, c2, f1 = self._parse_condition(condition)

        # timestep embedding -> cond vector
        t_emb = timestep_embedding(t, self.time_dim)
        cond = self.time_mlp(self.t_proj_in(t_emb))

        noisy_fea = self.encoder_noisy(noisy_fine2, cond=cond)
        f1_fea    = self.encoder_f1(f1, cond=cond)
        c2_fea    = self.encoder_coarse(c2, cond=cond)
        c1_fea    = self.encoder_coarse(c1, cond=cond)

        # stage-wise inject f1 into noisy features
        fused_fea = tuple(a + b for a, b in zip(noisy_fea, f1_fea))

        # keep your original (F, C2, C1) fusion in decoder
        out = self.decoder(fused_fea, c2_fea, c1_fea, cond=cond)
        return out

# -----------------------------
# 1) timestep embedding (DiT/ADM 常用的 sincos)
# -----------------------------
def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    if timesteps.dim() != 1:
        timesteps = timesteps.view(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb  # (B, dim)

class TimeMLP(nn.Module):
    def __init__(self, time_dim: int, cond_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
    def forward(self, t_emb):
        return self.net(t_emb)

# -----------------------------
# 2) AdaLN-Zero 风格的“安全调制”（stage-level）
#    - scale/shift/gate 全由 cond 向量产生
#    - gate / scale / shift 的最后线性层零初始化 => 初始几乎不改变主干
# -----------------------------
class AdaLNZeroStage(nn.Module):
    """
    x: (B, L, C)
    cond: (B, D)
    输出: x + gate * ( (1+scale)*LN(x) + shift )
    gate/scale/shift 的 head 零初始化，初始≈不影响主干。
    """
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.to_scale = nn.Linear(cond_dim, dim, bias=True)
        self.to_shift = nn.Linear(cond_dim, dim, bias=True)
        self.to_gate  = nn.Linear(cond_dim, dim, bias=True)

        nn.init.zeros_(self.to_scale.weight); nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight); nn.init.zeros_(self.to_shift.bias)
        nn.init.zeros_(self.to_gate.weight);  nn.init.zeros_(self.to_gate.bias)

    def forward(self, x, cond):
        # cond -> (B,1,C)
        scale = self.to_scale(cond).unsqueeze(1)
        shift = self.to_shift(cond).unsqueeze(1)
        gate  = self.to_gate(cond).unsqueeze(1)

        h = self.norm(x)
        h = (1.0 + scale) * h + shift
        return x + gate * h

# -----------------------------
# 3) Control/Adapter-lite：把 (c1,c2,f1) 编成多尺度 token residual
#    - residual heads 全部 zero-init => 初始不影响主干
# -----------------------------
class CondAdapterLite(nn.Module):
    """
    Align condition pyramid to token grids of PatchEmbed/PatchMerging.

    Input: c1,c2,f1 -> (B,6,H,W), H=img_size
    Output residual tokens:
      r0: (B, (H/down_scale)^2,   C)
      r1: (B, (H/(2*down_scale))^2,   C)
      r2: (B, (H/(4*down_scale))^2, 2C)
      r3: (B, (H/(8*down_scale))^2, 4C)
      r4: (B, (H/(16*down_scale))^2,8C)
    """
    def __init__(self, img_size: int, base_dim: int, channels: int = 6, down_scale: int = 2):
        super().__init__()
        assert img_size % down_scale == 0, "img_size must be divisible by down_scale"
        self.img_size = img_size
        self.down_scale = down_scale
        self.base_dim = base_dim
        in_ch = channels * 3

        # full-res stem (H, W)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_dim, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(base_dim, base_dim, 3, 1, 1),
            nn.SiLU(),
        )

        # align to PatchEmbed grid: (H/down_scale, W/down_scale)
        # use strided conv for learnable downsampling
        self.to_patch_grid = nn.Conv2d(base_dim, base_dim, kernel_size=down_scale, stride=down_scale, padding=0)

        # further downsamples by 2 to match PatchMerging levels
        self.down1 = nn.Conv2d(base_dim, base_dim, 4, 2, 1)         # -> H0/2
        self.down2 = nn.Conv2d(base_dim, base_dim * 2, 4, 2, 1)     # -> H0/4
        self.down3 = nn.Conv2d(base_dim * 2, base_dim * 4, 4, 2, 1) # -> H0/8
        self.down4 = nn.Conv2d(base_dim * 4, base_dim * 8, 4, 2, 1) # -> H0/16

        # zero-init output projections (ControlNet-style safety)
        self.out0 = nn.Conv2d(base_dim,     base_dim,     1, 1, 0)
        self.out1 = nn.Conv2d(base_dim,     base_dim,     1, 1, 0)
        self.out2 = nn.Conv2d(base_dim * 2, base_dim * 2, 1, 1, 0)
        self.out3 = nn.Conv2d(base_dim * 4, base_dim * 4, 1, 1, 0)
        self.out4 = nn.Conv2d(base_dim * 8, base_dim * 8, 1, 1, 0)

        for m in [self.out0, self.out1, self.out2, self.out3, self.out4]:
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def _to_tokens(x: torch.Tensor):
        b, c, h, w = x.shape
        return x.flatten(2).transpose(1, 2).contiguous()

    def forward(self, c1, c2, f1):
        x = torch.cat([c1, c2, f1], dim=1)  # (B,18,H,W)
        x = self.stem(x)                    # (B,C,H,W)

        # stage0 aligned to PatchEmbed grid
        x0 = self.to_patch_grid(x)          # (B,C,H0,W0), H0=H/down_scale

        x1 = self.down1(x0)                 # (B,C,  H0/2,  W0/2)
        x2 = self.down2(x1)                 # (B,2C, H0/4,  W0/4)
        x3 = self.down3(x2)                 # (B,4C, H0/8,  W0/8)
        x4 = self.down4(x3)                 # (B,8C, H0/16, W0/16)

        r0 = self._to_tokens(self.out0(x0))
        r1 = self._to_tokens(self.out1(x1))
        r2 = self._to_tokens(self.out2(x2))
        r3 = self._to_tokens(self.out3(x3))
        r4 = self._to_tokens(self.out4(x4))
        return r0, r1, r2, r3, r4

# -----------------------------
# 4) 你原 Encoder/Decoder 写死了 img_size=256，这里给出“参数化版本”
#    只做必要修改：PatchEmbed 的 img_size、各 stage resolution
# -----------------------------
class EncoderParam(nn.Module):
    def __init__(self, *, img_size: int, down_scale: int, in_dim: int, depths, agent_num, in_channels: int):
        super().__init__()
        self.img_size = img_size
        self.down_scale = down_scale
        self.in_dim = in_dim

        # 你已有 PatchEmbed
        self.inc = PatchEmbed(img_size=img_size, patch_size=down_scale, in_chans=in_channels,
                              embed_dim=in_dim, norm_layer=nn.LayerNorm)

        r0 = img_size // down_scale
        r1 = (img_size // 2) // down_scale
        r2 = (img_size // 4) // down_scale
        r3 = (img_size // 8) // down_scale

        self.enc1 = EncoderBlock(in_channels=in_dim, out_channels=in_dim, resolution=r0,
                                 downsample=PatchMerging, cur_depth=depths[0], agent_num=agent_num[0])
        self.enc2 = EncoderBlock(in_channels=in_dim, out_channels=in_dim * 2, resolution=r1,
                                 downsample=PatchMerging, cur_depth=depths[1], agent_num=agent_num[1])
        self.enc3 = EncoderBlock(in_channels=in_dim * 2, out_channels=in_dim * 4, resolution=r2,
                                 downsample=PatchMerging, cur_depth=depths[2], agent_num=agent_num[2])
        self.enc4 = EncoderBlock(in_channels=in_dim * 4, out_channels=in_dim * 8, resolution=r3,
                                 downsample=PatchMerging, cur_depth=depths[3], agent_num=agent_num[3])

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)
        return x1, x2, x3, x4, x5

class DecoderParam(nn.Module):
    def __init__(self, *, img_size: int, in_dim: int, down_scale: int, depths, agent_num, out_channels: int):
        super().__init__()
        self.img_size = img_size
        self.down_scale = down_scale

        r3 = (img_size // 8) // down_scale
        r2 = (img_size // 4) // down_scale
        r1 = (img_size // 2) // down_scale
        r0 = (img_size // 1) // down_scale
        r4 = (img_size // 16) // down_scale

        self.dec1 = DecoderBlock(in_dim * 8, in_dim * 4, r3, depths[3], agent_num=agent_num[3])
        self.dec2 = DecoderBlock(in_dim * 4, in_dim * 2, r2, depths[2], agent_num=agent_num[2])
        self.dec3 = DecoderBlock(in_dim * 2, in_dim,     r1, depths[1], agent_num=agent_num[1])
        self.dec4 = DecoderBlock(in_dim,     in_dim,     r0, depths[0], agent_num=agent_num[0])

        # x0-pred：不要 Tanh（推理时由 diffusion 里 clip_denoised 控制）
        self.outc = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(in_dim, out_channels, 3, 1, 1),
        )

        self.layer = SDFLayer(dim=in_dim * 8, input_resolution=(r4, r4),
                              depth=2, num_heads=max(1, (in_dim * 8) // 32), window_size=8,
                              mlp_ratio=4, qkv_bias=True, qk_scale=None,
                              drop=0.0, attn_drop=0.0, drop_path=0.,
                              norm_layer=nn.LayerNorm, agent_num=agent_num[3])

    def forward(self, fine_fea, coarse_fea2, coarse_fea1):
        x0 = self.layer(fine_fea[4], coarse_fea2[4], coarse_fea1[4])
        x1 = self.dec1(fine_fea[3], coarse_fea2[3], coarse_fea1[3], x0)
        x2 = self.dec2(fine_fea[2], coarse_fea2[2], coarse_fea1[2], x1)
        x3 = self.dec3(fine_fea[1], coarse_fea2[1], coarse_fea1[1], x2)
        x4 = self.dec4(fine_fea[0], coarse_fea2[0], coarse_fea1[0], x3)

        B, L, C = x4.shape
        x4 = x4.transpose(1, 2).view(B, C, self.img_size // self.down_scale, self.img_size // self.down_scale)
        return self.outc(x4)

# -----------------------------
# 5) 最终 Denoiser：接口与你 Diffusion 完全一致
#    - 自动 split condition (B,18,H,W) -> (c1,c2,f1)
#    - 时间条件 -> cond 向量
#    - 两个增强模块（都可开关 & zero-init 安全）
# -----------------------------
class FSDFormerDenoiserCVPR(nn.Module):
    """
    model(noisy_fine2, t, condition) -> x0_hat
    condition:
      - Tensor (B,18,H,W) = cat([c1,c2,f1])
      - tuple/list (c1,c2,f1)
    """
    def __init__(
        self,
        *,
        img_size: int,
        channels: int = 6,
        in_dim: int = 64,
        down_scale: int = 2,
        depths=(2, 2, 6, 2),
        agent_num=(9, 16, 64, 64),
        time_cond_dim: int = 256,
        enable_stage_adaln: bool = True,
        enable_cond_adapter: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.channels = channels
        self.enable_stage_adaln = enable_stage_adaln
        self.enable_cond_adapter = enable_cond_adapter

        # 时间条件向量
        self.time_dim = time_cond_dim
        self.time_mlp = TimeMLP(time_cond_dim, time_cond_dim)

        # 三个编码器：noisy / coarse(shared) / f1
        self.enc_noisy  = EncoderParam(img_size=img_size, down_scale=down_scale, in_dim=in_dim,
                                       depths=depths, agent_num=agent_num, in_channels=channels)
        self.enc_f1     = EncoderParam(img_size=img_size, down_scale=down_scale, in_dim=in_dim,
                                       depths=depths, agent_num=agent_num, in_channels=channels)
        self.enc_coarse = EncoderParam(img_size=img_size, down_scale=down_scale, in_dim=in_dim,
                                       depths=depths, agent_num=agent_num, in_channels=channels)

        self.dec = DecoderParam(img_size=img_size, in_dim=in_dim, down_scale=down_scale,
                                depths=depths, agent_num=agent_num, out_channels=channels)

        # stage-level AdaLN-Zero（5 个 stage：x1..x5）
        if enable_stage_adaln:
            self.adaln0 = AdaLNZeroStage(in_dim,     time_cond_dim)
            self.adaln1 = AdaLNZeroStage(in_dim,     time_cond_dim)
            self.adaln2 = AdaLNZeroStage(in_dim * 2, time_cond_dim)
            self.adaln3 = AdaLNZeroStage(in_dim * 4, time_cond_dim)
            self.adaln4 = AdaLNZeroStage(in_dim * 8, time_cond_dim)

        # Control/Adapter-lite：把条件做成多尺度 residual token
        if enable_cond_adapter:
            self.cond_adapter = CondAdapterLite(
                img_size=img_size,
                base_dim=in_dim,
                channels=channels,
                down_scale=down_scale
            )

        # f1 注入到 noisy 输入：1x1 residual，zero-init（更稳）
        self.f1_in_proj = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        nn.init.zeros_(self.f1_in_proj.weight)
        nn.init.zeros_(self.f1_in_proj.bias)

    def _parse_condition(self, condition):
        if isinstance(condition, (tuple, list)):
            assert len(condition) == 3
            return condition[0], condition[1], condition[2]
        assert torch.is_tensor(condition), "condition must be tuple/list or Tensor"
        assert condition.shape[1] == self.channels * 3, f"expect condition channels={self.channels*3}"
        c1, c2, f1 = torch.split(condition, self.channels, dim=1)
        return c1, c2, f1

    def _make_cond_vec(self, t: torch.Tensor):
        t_emb = timestep_embedding(t, self.time_dim)     # (B, time_dim)
        cond  = self.time_mlp(t_emb)                     # (B, time_dim)
        return cond

    def _stage_adaln(self, feats, cond):
        # feats: tuple(x1..x5)
        x1, x2, x3, x4, x5 = feats
        x1 = self.adaln0(x1, cond)
        x2 = self.adaln1(x2, cond)
        x3 = self.adaln2(x3, cond)
        x4 = self.adaln3(x4, cond)
        x5 = self.adaln4(x5, cond)
        return (x1, x2, x3, x4, x5)

    def forward(self, noisy_fine2, t, condition):
        """
        noisy_fine2: (B,6,H,W)
        t: (B,)
        condition: (B,18,H,W) or (c1,c2,f1)
        return: x0_hat (B,6,H,W)
        """
        c1, c2, f1 = self._parse_condition(condition)
        cond = self._make_cond_vec(t)

        # 先把 f1 做一个“安全的 residual hint”注入 noisy（不会改变通道数）
        noisy_in = noisy_fine2 + self.f1_in_proj(f1)

        # 编码三路
        fea_noisy = self.enc_noisy(noisy_in)
        fea_f1    = self.enc_f1(f1)
        fea_c2    = self.enc_coarse(c2)
        fea_c1    = self.enc_coarse(c1)

        # 条件侧路 residual（可选，且 zero-init，初始≈0）
        if self.enable_cond_adapter:
            r0, r1, r2, r3, r4 = self.cond_adapter(c1, c2, f1)
            fea_noisy = (
                fea_noisy[0] + r0,
                fea_noisy[1] + r1,
                fea_noisy[2] + r2,
                fea_noisy[3] + r3,
                fea_noisy[4] + r4,
            )

        # f1 逐尺度注入（时空一致性更强）
        fea_fused = tuple(a + b for a, b in zip(fea_noisy, fea_f1))

        # stage-level AdaLN-Zero（可选，zero-init，初始≈不影响）
        if self.enable_stage_adaln:
            fea_fused = self._stage_adaln(fea_fused, cond)
            fea_c2    = self._stage_adaln(fea_c2, cond)
            fea_c1    = self._stage_adaln(fea_c1, cond)

        # 解码（保持你原有 SDF 差分融合逻辑：F, C2, C1）
        x0_hat = self.dec(fea_fused, fea_c2, fea_c1)
        return x0_hat

# -------------------------
# Quick test
# -------------------------
if __name__ == '__main__':
    torch.manual_seed(0)

    B = 1
    H = W = 64
    noisy = torch.randn(B, 6, H, W)
    coarse1 = torch.randn(B, 6, H, W)  # example: same as original 6-ch
    coarse2 = torch.randn(B, 6, H, W)
    fine_1 = torch.randn(B, 6, H, W)
    t = torch.randint(low=0, high=1000, size=(B,))

    # If your coarse inputs are 6ch, and you concat noisy(3)+fine_1(3)=6ch, set coarse_channels=6 (or None if same)
    model = FSDFormerDenoiserCVPR(img_size=64, channels=6, enable_stage_adaln=True, enable_cond_adapter=True)

    y = model(noisy, t, (coarse1, coarse2, fine_1))
    print("output:", y.shape)  # (B, 3, 256, 256)
