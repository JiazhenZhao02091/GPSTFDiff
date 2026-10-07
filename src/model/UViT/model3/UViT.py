import torch
import torch.nn as nn
import math
from src.model.UViT.libs.timm_cy import trunc_normal_, Mlp
import einops
import torch.utils.checkpoint
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU, FP32SiLU, SwiGLU
from diffusers.utils import deprecate, logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch.nn.functional as F
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero, RMSNorm, SD35AdaLayerNormZeroX


if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

def patchify(imgs, patch_size):
    x = einops.rearrange(imgs, 'B C (h p1) (w p2) -> B (h w) (p1 p2 C)', p1=patch_size, p2=patch_size)
    return x

def unpatchify(x, channels=3):
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
    x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
    return x

def create_custom_mask(N_t, N_c, N_x):
    N = N_t + N_c + N_x

    mask = torch.zeros((N, N))

    start_c = N_t
    end_c = N_t + N_c
    start_x = end_c


    if start_c > 0:
        mask[start_c:end_c, :start_c] = float('-inf')


    mask[start_c:end_c, start_x:] = float('-inf')


    return mask


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FeedForwardControl(nn.Module):

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
        elif activation_fn == "swiglu":
            act_fn = SwiGLU(dim, inner_dim, bias=bias)

        self.net = nn.ModuleList([])

        self.net.append(act_fn)

        self.net.append(nn.Dropout(dropout))

        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))

        self.control_conv = zero_module(nn.Conv2d(inner_dim, inner_dim, 3, stride=1, padding=1, groups=inner_dim))

        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:

        for i, module in enumerate(self.net):
            hidden_states = module(hidden_states)
            if i == 1:
                timestep, patch_tokens = hidden_states[:, :1, :], hidden_states[:, 1:, :]
                hidden_states_control_org, hidden_states  = patch_tokens.chunk(2, dim=1)
                B, N, C = hidden_states.shape
                h = w = int(np.sqrt(N))
                assert h * w == N
                hidden_states_control = hidden_states_control_org.reshape(B, h, w, C).permute(0, 3, 1, 2)
                hidden_states_control = self.control_conv(hidden_states_control)
                hidden_states_control = hidden_states_control.reshape(B, C, N).permute(0, 2, 1)
                hidden_states = hidden_states + 1.2 * hidden_states_control
                hidden_states = torch.cat([timestep, hidden_states_control_org, hidden_states], dim=1)


        return hidden_states


class CMAAA(nn.Module):

    def __init__(self, dim, num_heads=8, pan_channel=None, ms_channel=None,
                 pan_ks=3, ms_ks=3, ka=3, qkv_bias=False, qk_norm=False,
                 attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()


        self.pan_channel = dim if pan_channel is None else pan_channel
        self.ms_channel = dim if ms_channel is None else ms_channel

        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.ka = ka


        self.dep_conv = nn.Conv2d(self.head_dim, self.ka * self.ka * self.head_dim, kernel_size=self.ka,
                                  bias=True, groups=self.head_dim, padding=self.ka // 2)


        self.q = nn.Conv2d(dim + self.ms_channel, dim, kernel_size=pan_ks, padding=pan_ks//2, bias=qkv_bias)
        self.k_pan = nn.Conv2d(dim + self.pan_channel, dim, kernel_size=pan_ks, padding=pan_ks//2, bias=qkv_bias)


        self.v_pan = nn.Conv2d(dim + self.pan_channel, dim, kernel_size=pan_ks, padding=pan_ks//2, bias=qkv_bias)
        self.kv_ms = nn.Conv2d(dim + self.ms_channel, dim * 2, kernel_size=ms_ks, padding=ms_ks//2, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm_pan = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm_ms = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.proj_pan = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.proj_ms = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm_time = AdaLayerNormZero(dim)

        self.reset_parameters()

    def reset_parameters(self):

        kernel = torch.zeros(self.ka * self.ka, self.ka, self.ka)
        for i in range(self.ka * self.ka):
            kernel[i, i // self.ka, i % self.ka] = 1.
        kernel = kernel.unsqueeze(1).repeat(self.head_dim, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, joint_features, H, W, s=None):
        B, L_total, C = joint_features.shape
        L = H * W
        assert L_total == 2 * L + 1, "Input length must be 2 * H * W + 1"

        timestep, patch_tokens = joint_features[:, :1, :], joint_features[:, 1:, :]
        cond_flat, x_flat  = patch_tokens.chunk(2, dim=1)

        x_flat, *_= self.norm_time(x_flat, emb=timestep.squeeze(1))


        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        cond = cond_flat.transpose(1, 2).reshape(B, C, H, W)


        q = self.q(torch.cat((x, cond), dim=1)).reshape(B, self.num_heads, self.head_dim, H, W)


        k_pan = self.k_pan(torch.cat((x, cond), dim=1)).reshape(B, self.num_heads, self.head_dim, H, W)


        v_pan = self.v_pan(torch.cat((x, cond), dim=1)).reshape(B, self.num_heads, self.head_dim, H, W)


        kv_ms = self.kv_ms(torch.cat((x, cond), dim=1)).reshape(B, 2, self.num_heads, self.head_dim, H, W)
        k_ms, v_ms = kv_ms[:, 0], kv_ms[:, 1]


        q, k_pan, k_pan = self.q_norm(q.permute(0, 1, 3, 4, 2)), self.k_norm_pan(k_pan.permute(0, 1, 3, 4, 2)), self.k_norm_ms(k_ms.permute(0, 1, 3, 4, 2))
        q, k_pan, k_ms = q.permute(0, 1, 4, 2, 3), k_pan.permute(0, 1, 4, 2, 3), k_ms.permute(0, 1, 4, 2, 3)

        k_pan, v_pan = k_pan.reshape(-1, self.head_dim, H, W), v_pan.reshape(-1, self.head_dim, H, W)
        k_ms, v_ms = k_ms.reshape(-1, self.head_dim, H, W), v_ms.reshape(-1, self.head_dim, H, W)

        q = q.reshape(B, self.num_heads, self.head_dim, 1, H, W) * self.scale

        k_pan = self.dep_conv(k_pan).reshape(B, self.num_heads, self.head_dim, self.ka * self.ka, H, W)
        v_pan = self.dep_conv(v_pan).reshape(B, self.num_heads, self.head_dim, self.ka * self.ka, H, W)
        k_ms = self.dep_conv(k_ms).reshape(B, self.num_heads, self.head_dim, self.ka * self.ka, H, W)
        v_ms = self.dep_conv(v_ms).reshape(B, self.num_heads, self.head_dim, self.ka * self.ka, H, W)

        k = torch.stack([k_pan, k_ms], dim=1)
        v = torch.stack([v_pan, v_ms], dim=1)

        attn = (q.unsqueeze(1) * k).sum(dim=3, keepdim=True).softmax(dim=4)
        attn = self.attn_drop(attn)

        x_out = (attn * v).sum(dim=4).reshape(B, 2, self.num_heads * self.head_dim, H, W)
        x_pan = self.proj_drop(self.proj_pan(x_out[:, 0]))
        x_ms = self.proj_drop(self.proj_ms(x_out[:, 1]))


        cond_flat = x_pan.flatten(2).transpose(1, 2)
        x_flat = x_ms.flatten(2).transpose(1, 2)


        out = torch.cat([timestep, cond_flat, x_flat], dim=1)

        return out


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False, grid_size=None, use_CMAAA=True, use_local_mlp=True):
        super().__init__()


        self.grid_size = grid_size
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)

        if use_local_mlp:
            self.mlp = FeedForwardControl(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        else:
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

        self.use_CMAAA = use_CMAAA
        if use_CMAAA:
            self.CM3A = CMAAA(dim=dim, num_heads=num_heads, norm_layer=norm_layer)

    def forward(self, x, skip=None, mask=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip, mask)
        else:
            return self._forward(x, skip, mask)

    def _forward(self, x, skip=None, mask=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x), attn_mask=mask)
        if self.use_CMAAA:
            x = x + self.CM3A(joint_features = self.norm3(x), H=self.grid_size, W=self.grid_size)
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class UViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1,
                 use_checkpoint=False, conv=True, skip=True, has_img=True, use_mask=False, use_CMAAA=True, use_local_mlp=True):
        super().__init__()

        print(f"use_CMAAA is {use_CMAAA}, use_loacl_mlp is {use_local_mlp}")
        self.num_features = self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.has_img = has_img

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.cond_init_conv = nn.Conv2d(in_chans * 3, in_chans, 3, padding=1)
        self.cond_embed  = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = (img_size // patch_size) ** 2

        self.grid_size = img_size // patch_size

        if use_mask:
            self.register_buffer('mask', create_custom_mask(N_t = 1, N_c  = num_patches, N_x = num_patches))
        else:
            self.mask = None

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()


        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, embed_dim)
            self.extras = 2
        else:
            self.extras = 1

        if self.has_img:
            self.extras += num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, embed_dim))

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint, grid_size = self.grid_size, use_CMAAA=use_CMAAA, use_local_mlp=use_local_mlp)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint, grid_size = self.grid_size, use_CMAAA=use_CMAAA, use_local_mlp=use_local_mlp)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint, grid_size = self.grid_size, use_CMAAA=use_CMAAA, use_local_mlp=use_local_mlp)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        self.patch_dim = patch_size ** 2 * in_chans
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)
        self.final_layer = nn.Conv2d(self.in_chans, self.in_chans, 3, padding=1) if conv else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x, timesteps, y=None):
        x = self.patch_embed(x)
        B, L, D = x.shape

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)

        if y is not None:
            cond_img = self.cond_init_conv(y)
            cond_img = self.cond_embed(cond_img)
            x = torch.cat((cond_img, x), dim=1)

        x = torch.cat((time_token, x), dim=1)
        x = x + self.pos_embed

        skips = []
        for blk in self.in_blocks:
            x = blk(x, mask=self.mask)
            skips.append(x)

        x = self.mid_block(x, mask=self.mask)

        for blk in self.out_blocks:
            x = blk(x, skip=skips.pop(), mask=self.mask)

        x = self.norm(x)
        x = self.decoder_pred(x)
        assert x.size(1) == self.extras + L
        x = x[:, self.extras:, :]
        x = unpatchify(x, self.in_chans)
        x = self.final_layer(x)
        return x

if __name__ == '__main__':
    device = "cuda"
    model = UViT(
        img_size=256,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,

    )
    model = model.to(device)


    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    pass
