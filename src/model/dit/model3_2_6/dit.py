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
    # x: (B, N, n_heads, head_dim) -> reshaped for broadcasting
    # freqs_cis: (N, head_dim/2) complex tensor
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, -1) # (1, N, 1, head_dim/2)
    x_out = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    return x_out.type_as(x)

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

        self.wa = WaveletAttention(channels=hidden_size)

    def forward(self, x, c_t, freqs_cis=None): # c_img = None
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_mlp2, scale_mlp2, gate_mlp2 = self.adaLN_modulation(c_t).chunk(9, dim=1)
       
        x = x + self.drop_path(gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), freqs_cis=freqs_cis))
       
        x = x + self.drop_path(gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        # Feed Fself.drop_path(orward Control
        x = x + self.drop_path(gate_mlp2.unsqueeze(1) * self.mlp_2(modulate(self.norm_ffc(x), shift_mlp2, scale_mlp2)))
        
        # 2. 加入 WA 增强空间/频域特征
        # 注意：由于你的 x 是 cat([x, condition]) 得到的，长度是 2*N
        # 我们需要分别对 x 和 condition 进行处理，或者只处理前半部分
        B, Total_N, C = x.shape
        N = Total_N // 2
        h = w = int(np.sqrt(N))

        x_main, x_cond = x.chunk(2, dim=1) # 拆回两个图像的 token 序列

        def apply_wa(feat):
            f_2d = feat.transpose(1, 2).reshape(B, C, h, w)
            f_2d = self.wa(f_2d)
            return f_2d.reshape(B, C, N).transpose(1, 2)

        x_main = apply_wa(x_main)
        x_cond = apply_wa(x_cond) # condition 也可以选择是否应用

        x = torch.cat([x_main, x_cond], dim=1)
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

    def forward(self, x, t, condition):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        c1, c2 ,f1 = condition.split(6, dim=1)
        # Pre
        coarse = torch.cat([c1, c2], dim=1)
        coarse = self.init_coarse(coarse)
        fine = self.init_fine(f1)
        condition = torch.cat([coarse, fine], dim=1)
        assert condition.shape[1] == self.in_channels * 2, "condtion的序列长度必须为2 * inchannels"
        condition = self.init_cond(condition)
        x = self.init_x(x)
        
        # Embedding
        # x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        # condition = self.cond_embedder(condition) + self.cond_pos_embed
        # t = self.t_embedder(t) 

        # Embedding (No more + self.pos_embed)
        x = self.x_embedder(x) 
        condition = self.cond_embedder(condition)
        t = self.t_embedder(t) 

        x = torch.cat([x, condition], dim = 1)  # x 在前 condtion 在后面
        assert x.shape[1] == 2 * condition.shape[1], "x的序列长度应为 2*L"

        # Prepare RoPE frequencies for current sequence length
        seq_len = x.shape[1]
        freqs_cis = self.freqs_cis[:seq_len]

        c_t = t
        for block in self.blocks:
            x = block(x, c_t, freqs_cis=freqs_cis)    

        # TODO后续forward逻辑
        x = self.final_layer(x, c_t)
        x, _ = x.chunk(2, dim=1)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x
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


def build_wavelet_kernels(device=None, dtype=torch.float32):                                                                                                                                                                     # 微信公众号:AI缝合术
    """
    返回 2x2 的四个 2D 分析核：LL, LH, HL, HH
    对于 Db1 (Haar)：h0=[1/sqrt2, 1/sqrt2], h1=[-1/sqrt2, 1/sqrt2]                                                                                                                                                                     # 微信公众号:AI缝合术
    2D 核是外积：h_row^T * h_col
    """
    s = 1.0 / math.sqrt(2.0)
    h0 = torch.tensor([s, s], dtype=dtype, device=device)      # 低通                                                                                                                                                                     # 微信公众号:AI缝合术
    h1 = torch.tensor([-s, s], dtype=dtype, device=device)     # 高频                                                                                                                                                                     # 微信公众号:AI缝合术
    # 外积得到 2x2 核
    LL = torch.ger(h0, h0)  # 低-低
    LH = torch.ger(h0, h1)  # 低-高（垂直边）
    HL = torch.ger(h1, h0)  # 高-低（水平边）
    HH = torch.ger(h1, h1)  # 高-高（对角）
    # 形状统一为 (1,1,2,2) 方便后续扩展到 groups=C
    filt = torch.stack([LL, LH, HL, HH], dim=0).unsqueeze(1)                                                                                                                                                                     # 微信公众号:AI缝合术
    return filt  # (4,1,2,2)

# =========================
# Wavelet Attention 模块
# =========================
class WaveletAttention(nn.Module):
    """
    实现步骤：
    X --DWT--> (LH, HL, HH, LL)
         高频阈值化 -> concat -> 1x1 conv 融合 -> 与 LL 做 IDWT -> X_re
         GAP -> (可选FC) -> Softmax -> 通道权重
         输出： Final = weight * X
    """
    def __init__(self, channels, use_fc=True):
        super().__init__()
        self.channels = channels
        self.use_fc = use_fc

        # 软阈值参数（3 个高频子带 * C），sigmoid 约束到 0~1，再乘以 mean(|x|)
        self.theta = nn.Parameter(torch.zeros(3, channels, 1, 1))
        # 高频子带融合：将 3C -> C
        self.fuse = nn.Conv2d(3 * channels, channels, kernel_size=1, bias=False)                                                                                                                                                                     # 微信公众号:AI缝合术
        # GAP 后可选的 FC（保持维度 C->C）
        if use_fc:
            self.fc = nn.Linear(channels, channels, bias=True)

        # 小波核（注册为 buffer，参与 to(device) 但不训练）
        filt = build_wavelet_kernels()
        self.register_buffer("w_analysis", filt)   # (4,1,2,2)
        self.register_buffer("w_synthesis", filt)  # Db1 正交：合成=分析

    # ---------- DWT 与 IDWT ----------
    def dwt(self, x):
        """
        x: (B,C,H,W)
        返回：LH, HL, HH, LL 以及中间 size 信息
        """
        B, C, H, W = x.shape

        # 零填充到偶数尺寸，避免边界丢失
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)                                                                                                                                                                     # 微信公众号:AI缝合术

        # 组卷积：每个通道使用同一组 4 个滤波器
        # 权重形状需要扩展为 (4*C, 1, 2, 2) 并 groups=C
        weight = self.w_analysis.repeat(C, 1, 1, 1)  # (4C,1,2,2)
        y = F.conv2d(x, weight=weight, bias=None, stride=2, padding=0, groups=C)  # (B,4C,H/2,W/2)                                                                                                                                                                     # 微信公众号:AI缝合术

        # 按子带拆分
        y = y.view(B, C, 4, y.size(-2), y.size(-1)).contiguous()                                                                                                                                                                     # 微信公众号:AI缝合术
        LL = y[:, :, 0]  # (B,C,h,w)
        LH = y[:, :, 1]
        HL = y[:, :, 2]
        HH = y[:, :, 3]
        return LH, HL, HH, LL

    def idwt(self, LH, HL, HH, LL):
        """
        逆变换：将四个子带重建为 (B,C,H,W)
        """
        B, C, h, w = LL.shape
        # 将 4 个子带 stack 回 (B,4C,h,w)
        y = torch.stack([LL, LH, HL, HH], dim=2).view(B, 4 * C, h, w)

        # conv_transpose2d 作为合成滤波器，stride=2
        weight = self.w_synthesis.repeat(C, 1, 1, 1)  # (4C,1,2,2)
        # conv_transpose 的权重形状：(in_channels, out_channels/groups, kH, kW)
        # 我们希望 groups=C，每组把 4 个子带合成为 1 个通道
        # 需要把 weight 视作 (4C, 1, 2, 2)，设置 groups=C 时会自动每4个输入映射到1个输出
        x_rec = F.conv_transpose2d(y, weight=weight, bias=None, stride=2, padding=0, groups=C)                                                                                                                                                                     # 微信公众号:AI缝合术
        return x_rec

    # ---------- 高频软阈值 ----------
    @staticmethod
    def soft_threshold(x, thr):
        # soft-shrinkage： sign(x) * relu(|x| - thr)
        return torch.sign(x) * F.relu(torch.abs(x) - thr)

    # ---------- 前向 ----------
    def forward(self, x):
        B, C, H, W = x.shape

        # 1) DWT
        LH, HL, HH, LL = self.dwt(x)

        # 2) 高频子带阈值化与融合
        # 归一化后的阈值（按通道），值域约束 0~1，再乘以该子带的平均幅度
        eps = 1e-6
        m_LH = LH.abs().mean(dim=(2, 3), keepdim=True) + eps
        m_HL = HL.abs().mean(dim=(2, 3), keepdim=True) + eps
        m_HH = HH.abs().mean(dim=(2, 3), keepdim=True) + eps

        t = torch.sigmoid(self.theta)  # (3,C,1,1)
        thr_LH = t[0].unsqueeze(0) * m_LH
        thr_HL = t[1].unsqueeze(0) * m_HL
        thr_HH = t[2].unsqueeze(0) * m_HH

        LH_hat = self.soft_threshold(LH, thr_LH)
        HL_hat = self.soft_threshold(HL, thr_HL)
        HH_hat = self.soft_threshold(HH, thr_HH)

        # 融合卷积（将 3C -> C）
        H_concat = torch.cat([LH_hat, HL_hat, HH_hat], dim=1)  # (B,3C,h,w)                                                                                                                                                                     # 微信公众号:AI缝合术
        H_fused = self.fuse(H_concat)  # (B,C,h,w)

        # 3) IDWT 重构
        X_re = self.idwt(LH_hat, HL_hat, H_fused, LL)  # (B,C,H',W')，H'/W'≈H/W                                                                                                                                                                     # 微信公众号:AI缝合术

        # 4) 注意力权重：GAP -> (可选FC) -> Softmax(沿通道)
        gap = F.adaptive_avg_pool2d(X_re, 1).view(B, C)  # (B,C)
        if self.use_fc:
            gap = self.fc(gap)  # (B,C)
        attn = F.softmax(gap, dim=1).view(B, C, 1, 1)  # (B,C,1,1)

        # 5) 加权原输入
        out = x * attn
        return out

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