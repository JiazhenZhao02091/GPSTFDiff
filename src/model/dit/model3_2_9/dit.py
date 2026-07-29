# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU, FP32SiLU, SwiGLU
from diffusers.utils import deprecate, logging
from typing import Any, Dict, List, Optional, Tuple
import torch.nn.functional as F
from timm.models.layers import DropPath

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


#################################################################################
#                                    ChangeMaskHead                             #
#################################################################################
class ChangeMaskHead(nn.Module):
    """
    输入：变化线索 (B, Cin, H, W)
    输出：变化概率 m (B, 1, H, W)，值越大表示越“应该用 c2 而不是 f1”
    """
    def __init__(self, in_ch: int, mid_ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(mid_ch, mid_ch, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(mid_ch, 1, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


#################################################################################
#                                    RMSNorm                                    #
#################################################################################
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed
    
#################################################################################
#                             RoPE Implementation                               #
#################################################################################
def apply_rotary_emb(x, freqs_cis):
    """
    x: (B, N, n_heads, head_dim)
    freqs_cis:
      - (N, head_dim/2)  或
      - (B, N, head_dim/2)
      dtype: complex
    """
    B, N, H, D = x.shape
    assert D % 2 == 0, "head_dim must be even for RoPE"

    x_complex = torch.view_as_complex(
        x.float().reshape(B, N, H, D // 2, 2)
    )  # (B, N, H, D/2) complex

    if freqs_cis.dim() == 2:
        # (N, D/2) -> (1, N, 1, D/2)
        freqs = freqs_cis[None, :, None, :]
    elif freqs_cis.dim() == 3:
        # (B, N, D/2) -> (B, N, 1, D/2)
        freqs = freqs_cis[:, :, None, :]
    else:
        raise ValueError(f"freqs_cis shape error: {freqs_cis.shape}")

    out = torch.view_as_real(x_complex * freqs).reshape(B, N, H, D)
    return out.type_as(x)


def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # (N, dim/2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

#################################################################################
#                                 Flash Attention                               #
#################################################################################
class FlashAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., **kwargs):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, freqs_cis=None): # Add freqs_cis argument
        B, N, C = x.shape
        # qkv: (B, N, 3, num_heads, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        
        # Split q, k, v
        q, k, v = qkv.unbind(2) # (B, N, num_heads, head_dim)

        # Apply RoPE if provided
        if freqs_cis is not None:
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)

        # Permute for Flash Attention: (B, num_heads, N, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

#################################################################################
#                                 Feed Forward Control                          #
#################################################################################

class FeedForwardControl(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

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
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        # zero convolution
        self.control_conv = zero_module(nn.Conv2d(inner_dim, inner_dim, 3, stride=1, padding=1, groups=inner_dim)) 
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # print(f"net len is {len(self.net)}")
        for i, module in enumerate(self.net):
            hidden_states = module(hidden_states)
            if i == 1:
                hidden_states, hidden_states_control_org = hidden_states.chunk(2, dim=1)
                B, N, C = hidden_states.shape
                h = w = int(np.sqrt(N))
                assert h * w == N
                hidden_states_control = hidden_states_control_org.reshape(B, h, w, C).permute(0, 3, 1, 2)
                hidden_states_control = self.control_conv(hidden_states_control)
                hidden_states_control = hidden_states_control.reshape(B, C, N).permute(0, 2, 1)
                hidden_states = hidden_states + 1.2 * hidden_states_control # TODO: add control signal, better change to 1.0 when training
                hidden_states = torch.cat([hidden_states, hidden_states_control_org], dim=1)
        return hidden_states

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0,  drop_path=0.0,**block_kwargs):
        super().__init__()

        # self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.attn = FlashAttention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=dropout,  proj_drop=dropout, **block_kwargs)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=dropout)
        self.norm_ffc = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)        
        self.mlp_2 = FeedForwardControl(dim=hidden_size, dim_out=hidden_size, activation_fn="gelu-approximate", dropout=dropout)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, c_t, freqs_cis=None): # c_img = None
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_mlp2, scale_mlp2, gate_mlp2 = self.adaLN_modulation(c_t).chunk(9, dim=1)
       
        x = x + self.drop_path(gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), freqs_cis=freqs_cis))
       
        x = x + self.drop_path(gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        # Feed Fself.drop_path(orward Control
        x = x + self.drop_path(gate_mlp2.unsqueeze(1) * self.mlp_2(modulate(self.norm_ffc(x), shift_mlp2, scale_mlp2)))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        # self.norm_final = RMSNorm(hidden_size, eps=1e-6)
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        # class_dropout_prob=0.1,
        # num_classes=1000,
        dropout=0.0,        # Dropout probability
        drop_path_rate=0.0, # Drop path rate
        learn_sigma=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # 额外：为 target coarse@t2 做一个语义分支（轻量 3x3）
        self.init_sem = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

        # change mask head：输入用两个变化线索：|c2-c1| 和 |f1-up(c1)|（你这里同分辨率就直接 |f1-c1|）
        # 每个都是 6 通道，所以 in_ch=12
        self.change_head = ChangeMaskHead(in_ch=in_channels * 2, mid_ch=32)

        # 把 m 做 patchify 成 token gate（用 avgpool 就行）
        self.mask_pool = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)

        # Unsed
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.cond_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.num_patches = self.x_embedder.num_patches
        # ---- Will use fixed sin-cos embedding: ----#
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        # self.cond_pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        # ---- Will use fixed ROPE embedding: ----#
        # Note: We double the length because we concat x and condition
        self.max_seq_len = self.num_patches * 2 
        freqs_cis = precompute_freqs_cis(self.head_dim, self.max_seq_len)
        self.register_buffer('freqs_cis', freqs_cis, persistent=False)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout=dropout, drop_path=dpr[i]) for i in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        # Cond
        self.init_coarse = nn.Conv2d(in_channels * 2, in_channels, 3, 1, 1)
        self.init_fine = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.init_cond = nn.Conv2d(in_channels * 2, in_channels, 1) # 2C -> C
        self.init_x  = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        
        self.initialize_weights() # 初始化

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize cond_embedder (类似 x_embedder):
        cond_w = self.cond_embedder.proj.weight.data
        nn.init.xavier_uniform_(cond_w.view([cond_w.shape[0], -1]))
        nn.init.constant_(self.cond_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # Initialize Conv2d layers (输入预处理层，使用 kaiming 初始化，适合后续的激活函数):
        nn.init.xavier_uniform_(self.init_x.weight)
        nn.init.constant_(self.init_x.bias, 0)

        nn.init.xavier_uniform_(self.init_fine.weight)
        nn.init.constant_(self.init_fine.bias, 0)

        nn.init.xavier_uniform_(self.init_coarse.weight)
        nn.init.constant_(self.init_coarse.bias, 0)

        nn.init.xavier_uniform_(self.init_cond.weight)
        nn.init.constant_(self.init_cond.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, condition, return_mask: bool = False):
        c1, c2, f1 = condition.split(6, dim=1)
        # ===== 1) 准备两路条件特征：语义(来自 c2) + 纹理(来自 f1) =====
        # 语义：更相信 target 时刻的 coarse@t2
        sem = self.init_sem(c2)  # (B,6,H,W)

        # 纹理：fine@t1
        tex = self.init_fine(f1) # (B,6,H,W)

        # ===== 2) 计算 change mask m =====
        # 变化线索（都很轻，不引入额外数据）
        d_coarse = torch.abs(c2 - c1)   # (B,6,H,W)
        d_cf     = torch.abs(f1 - c1)   # (B,6,H,W)  # 你这里同分辨率，直接减
        m = self.change_head(torch.cat([d_coarse, d_cf], dim=1))  # (B,1,H,W), in [0,1]

        # ===== 3) 变化感知混合：变化区用 sem，未变化区用 tex =====
        cond_mix_sem = sem * m
        cond_mix_tex = tex * (1.0 - m)
        cond = torch.cat([cond_mix_sem, cond_mix_tex], dim=1)     # (B,12,H,W)
        cond = self.init_cond(cond)                               # (B,6,H,W)

        # x 分支保持不变
        x = self.init_x(x)

        # ===== 4) Patch embed =====
        x_tokens = self.x_embedder(x)           # (B,L,D)
        cond_tokens = self.cond_embedder(cond)  # (B,L,D)
        t_emb = self.t_embedder(t)
        c_t = t_emb

        # patch-level gate（可选：用于 block 内更精细门控）
        m_tokens = self.mask_pool(m).flatten(2).transpose(1, 2)   # (B,L,1)

        # ===== 5) 走 transformer blocks（下面给你一个最小的“门控注入增强”写法）=====
        seq = torch.cat([x_tokens, cond_tokens], dim=1)

        # pos ids（你的 RoPE 仍然可用）
        B, L, _ = x_tokens.shape
        x_pos = torch.arange(L, device=x_tokens.device)
        pos_ids = torch.cat([x_pos, x_pos + L], dim=0).unsqueeze(0).expand(B, -1)  # (B,2L)
        freqs_cis = self.freqs_cis[pos_ids]  # (B,2L,head_dim/2)

        for block in self.blocks:
            seq = block(seq, c_t, freqs_cis=freqs_cis)

            # --- 最小门控增强（可选但很推荐）：把 cond_tokens 以 gate 注入 x_tokens ---
            # 解释：即便你用 concat self-attn，cond 对 x 的影响仍可能在深层被“稀释”，
            #      这个 residual 注入能显著增强“变化区听 c2，未变化区听 f1”的控制力。
            x_part, c_part = seq.split([L, L], dim=1)
            x_part = x_part + (m_tokens * c_part) * 0.05  # 0.1 是强度，可调
            seq = torch.cat([x_part, c_part], dim=1)

        # ===== 6) 输出 =====
        seq = self.final_layer(seq, c_t)
        out_x, _ = seq.chunk(2, dim=1)
        out_x = self.unpatchify(out_x)
        # return mask
        if return_mask:
            return out_x, m
        return out_x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

# def DiT_S_8(**kwargs):
#     return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, input_size=256, in_channels=6, learn_sigma=False, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}


if __name__ == "__main__":
    device = "cuda"
    model = DiT_models['DiT-S/8']().to(device)
    x = torch.randn(1, 6, 256, 256).to(device)
    condition = torch.randn(1, 18, 256, 256).to(device)
    t = torch.randint(0, 1000, (x.shape[0], ), device = device)

    outputs = model(x, t, condition)
    print("successfully load DiT-S/8")
    print(f"outputs.shape is {outputs.shape}")
    print(sum(p.numel() for p in model.parameters()))
    for p in model.parameters():
        print(p.dtype)
        break
    # 统计前向传播的FLOPs
    try:
        from thop import profile
        flops, params = profile(model, inputs=(x, t, condition), verbose=False)
        print(f"Total FLOPs (forward): {flops}, Params: {params}")
        print(f"FLOPs per sample: {flops / x.shape[0]}")
    except Exception as e:
        print("Unable to compute FLOPs using thop. Error:", e)