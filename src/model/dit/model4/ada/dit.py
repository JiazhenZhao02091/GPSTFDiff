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


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


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
#                                       CMAAA                                   #
#################################################################################
# 原始CMAAA
class CMAAA(nn.Module):
    def __init__(self, dim, num_heads=8, pan_channel=1, ms_channel=8, pan_ks=3, ms_ks=3, ka=3, qkv_bias=False, qk_norm=False,
                 attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.pan_channel = pan_channel
        self.ms_channel = ms_channel
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pan_ks = pan_ks
        self.ms_ks = ms_ks
        pan_pw = pan_ks // 2
        ms_pw = ms_ks // 2
        self.ka = ka

        self.dep_conv = nn.Conv2d(self.head_dim, self.ka * self.ka * self.head_dim, kernel_size=self.ka,
                                  bias=True, groups=self.head_dim, padding=self.ka // 2)

        self.q = nn.Conv2d(dim + ms_channel, dim, kernel_size=pan_ks, padding=pan_pw, bias=qkv_bias)
        self.k_pan = nn.Conv2d(dim + pan_channel, dim, kernel_size=pan_ks, padding=pan_pw, bias=qkv_bias)
        self.v_pan = nn.Conv2d(dim + pan_channel * 3, dim, kernel_size=pan_ks, padding=pan_pw, bias=qkv_bias)
        self.kv_ms = nn.Conv2d(dim + ms_channel, dim * 2, kernel_size=ms_ks, padding=ms_pw, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm_pan = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm_ms = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.proj_pan = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.proj_ms = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def reset_parameters(self):
        # shift initialization for group convolution
        kernel = torch.zeros(self.ka * self.ka, self.ka, self.ka)
        for i in range(self.ka * self.ka):
            kernel[i, i // self.ka, i % self.ka] = 1.
        kernel = kernel.unsqueeze(1).repeat(self.head_dim, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x, ms, lpan, pan, s):
        B, C, H, W = x.shape

        pan_ = lpan.repeat(1, self.ms_channel, 1, 1)
        cond = pan_ * (1.0 - s).view(-1, 1, 1, 1) + ms * s.view(-1, 1, 1, 1)
        q = self.q(torch.cat((x, cond), dim=1)).reshape(B, self.num_heads, self.head_dim, H, W)  # B, N, C, H, W
        k_pan = self.k_pan(torch.cat((x, lpan), dim=1)).reshape(B, self.num_heads, self.head_dim, H, W)
        v_pan = self.v_pan(torch.cat((x, lpan, pan, pan - lpan), dim=1)).reshape(B, self.num_heads, self.head_dim, H, W)
        kv_ms = self.kv_ms(torch.cat((x, ms), dim=1)).reshape(B, 2, self.num_heads, self.head_dim, H, W)
        k_ms, v_ms = kv_ms[:, 0], kv_ms[:, 1]  # B, N, C, H, W
        q, k_pan, k_pan = self.q_norm(q.permute(0, 1, 3, 4, 2)), self.k_norm_pan(k_pan.permute(0, 1, 3, 4, 2)), self.k_norm_ms(k_ms.permute(0, 1, 3, 4, 2))
        q, k_pan, k_ms = q.permute(0, 1, 4, 2, 3), k_pan.permute(0, 1, 4, 2, 3), k_ms.permute(0, 1, 4, 2, 3)  # B, N, C, H, W
        k_pan, v_pan = k_pan.reshape(-1, self.head_dim, H, W), v_pan.reshape(-1, self.head_dim, H, W)
        k_ms, v_ms = k_ms.reshape(-1, self.head_dim, H, W), v_ms.reshape(-1, self.head_dim, H, W)
        q = q.reshape(B, self.num_heads, self.head_dim, 1, H, W) * self.scale  # B, N, C, 1, H, W
        k_pan = self.dep_conv(k_pan).reshape(B, self.num_heads, self.head_dim, self.ka * self.ka, H, W)
        v_pan = self.dep_conv(v_pan).reshape(B, self.num_heads, self.head_dim, self.ka * self.ka, H, W)
        k_ms = self.dep_conv(k_ms).reshape(B, self.num_heads, self.head_dim, self.ka * self.ka, H, W)
        v_ms = self.dep_conv(v_ms).reshape(B, self.num_heads, self.head_dim, self.ka * self.ka, H, W)

        k = torch.stack([k_pan, k_ms], dim=1)
        v = torch.stack([v_pan, v_ms], dim=1)
        attn = (q.unsqueeze(1) * k).sum(dim=3, keepdim=True).softmax(dim=4)
        attn = self.attn_drop(attn)

        x = (attn * v).sum(dim=4).reshape(B, 2, self.num_heads * self.head_dim, H, W)
        x_pan = self.proj_drop(self.proj_pan(x[:, 0]))
        x_ms = self.proj_drop(self.proj_ms(x[:, 1]))
        return x_pan, x_ms
# 多分支注入的Block
class STF_CMAAA(nn.Module):
    def __init__(self, dim, num_heads=8, 
                 ref_channel=6,   # 参考高分影像通道数
                 tar_channel=6,   # 目标低分影像通道数
                 bottleneck_dim=60, # 【新增】瓶颈层维度，推荐 64 或 96
                 ref_ks=3, tar_ks=3, ka=3, 
                 qkv_bias=False, qk_norm=False,
                 attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm,
                 out_channels=None, patch_size=None):
        super().__init__()
        
        # 1. 基础参数检查
        # 注意：这里 num_heads 是针对 bottleneck_dim 的，而不是原始 dim
        # 如果 bottleneck_dim=64, num_heads=8, 则每个头 8 维
        assert bottleneck_dim % num_heads == 0, f'bottleneck_dim {bottleneck_dim} should be divisible by num_heads {num_heads}'
        
        self.dim = dim
        self.bottleneck_dim = bottleneck_dim
        self.num_heads = num_heads
        self.head_dim = bottleneck_dim // num_heads # 注意：head_dim 变小了
        self.scale = self.head_dim ** -0.5
        self.ka = ka
        self.ref_channel = ref_channel
        self.tar_channel = tar_channel

        # 2. 【关键】降维与升维投影层 (1x1 Conv)
        self.down_proj = nn.Conv2d(dim, bottleneck_dim, kernel_size=1)
        self.up_proj = nn.Conv2d(bottleneck_dim, dim, kernel_size=1)

        # 3. 核心卷积层 (现在全部基于 bottleneck_dim 构建，显存占用极小)
        ref_pw = ref_ks // 2
        tar_pw = tar_ks // 2
        
        self.dep_conv = nn.Conv2d(self.head_dim, self.ka * self.ka * self.head_dim, kernel_size=self.ka,
                                  bias=True, groups=self.head_dim, padding=self.ka // 2)

        # Q, K, V 的输入输出都变小了
        # 输入: bottleneck_dim + 6 -> 输出: bottleneck_dim
        self.q = nn.Conv2d(bottleneck_dim + tar_channel, bottleneck_dim, kernel_size=ref_ks, padding=ref_pw, bias=qkv_bias)
        self.k_ref = nn.Conv2d(bottleneck_dim + ref_channel, bottleneck_dim, kernel_size=ref_ks, padding=ref_pw, bias=qkv_bias)
        self.v_ref = nn.Conv2d(bottleneck_dim + ref_channel * 3, bottleneck_dim, kernel_size=ref_ks, padding=ref_pw, bias=qkv_bias)
        self.kv_tar = nn.Conv2d(bottleneck_dim + tar_channel, bottleneck_dim * 2, kernel_size=tar_ks, padding=tar_pw, bias=qkv_bias)

        # Norms (针对 head_dim)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm_ref = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm_tar = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        # Attention 内部的融合投影
        self.inner_proj = nn.Conv2d(bottleneck_dim * 2, bottleneck_dim, kernel_size=1, bias=True)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # 4. 外部 Pre-Norm (针对原始维度)
        # self.norm = norm_layer(dim)

        self.reset_parameters()

    def reset_parameters(self):
        # Shift Conv Init
        kernel = torch.zeros(self.ka * self.ka, self.ka, self.ka)
        for i in range(self.ka * self.ka):
            kernel[i, i // self.ka, i % self.ka] = 1.
        kernel = kernel.unsqueeze(1).repeat(self.head_dim, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=False)
        
        # 【关键】零初始化最后的升维层 (Up-Projection)
        # 确保初始状态下模块输出为 0，不影响主干
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, tar_coarse, ref_coarse, ref_fine):
        # x: [B, 2*L, D]
        # 1. Pre-Norm (在 384 维度上做)
        # x_norm = self.norm(x)
        x_norm = x # norm 在外部进行
        x_tokens, _ = x_norm.chunk(2, dim=1) 
        
        B, N, C = x_tokens.shape 
        h = w = int(N ** 0.5)

        # 2. 特征重塑 [B, 384, h, w]
        x_small = x_tokens.transpose(1, 2).reshape(B, C, h, w)

        # ==================== 步骤 A: 通道压缩 ====================
        # [B, 384, h, w] -> [B, 64, h, w]
        # 此时显存占用大大减小
        x_reduced = self.down_proj(x_small) 
        
        # ==================== 步骤 B: 空间上采样 ====================
        # [B, 64, h, w] -> [B, 64, 256, 256]
        # 虽然分辨率变大了，但因为通道只有 64，显存压力可控
        target_H, target_W = tar_coarse.shape[-2:]
        x_feat_large = F.interpolate(x_reduced, size=(target_H, target_W), mode='bilinear', align_corners=False)

        # ==================== 步骤 C: CMAAA (Bottleneck Dim) ====================
        H, W = target_H, target_W
        
        # 注意：这里拼接的是 x_reduced (64通道) + tar_coarse (6通道) = 70通道
        q = self.q(torch.cat((x_feat_large, tar_coarse), dim=1)).reshape(B, self.num_heads, self.head_dim, H, W)
        
        k_ref = self.k_ref(torch.cat((x_feat_large, ref_coarse), dim=1)).reshape(B, self.num_heads, self.head_dim, H, W)
        residuals = ref_fine - ref_coarse
        v_ref = self.v_ref(torch.cat((x_feat_large, ref_coarse, ref_fine, residuals), dim=1)).reshape(B, self.num_heads, self.head_dim, H, W)
        
        kv_tar = self.kv_tar(torch.cat((x_feat_large, tar_coarse), dim=1)).reshape(B, 2, self.num_heads, self.head_dim, H, W)
        k_tar, v_tar = kv_tar[:, 0], kv_tar[:, 1]

        q = self.q_norm(q.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3) * self.scale
        k_ref = self.k_norm_ref(k_ref.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        k_tar = self.k_norm_tar(k_tar.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        
        k_ref = self.dep_conv(k_ref.reshape(-1, self.head_dim, H, W)).reshape(B, self.num_heads, self.head_dim, -1, H, W)
        v_ref = self.dep_conv(v_ref.reshape(-1, self.head_dim, H, W)).reshape(B, self.num_heads, self.head_dim, -1, H, W)
        k_tar = self.dep_conv(k_tar.reshape(-1, self.head_dim, H, W)).reshape(B, self.num_heads, self.head_dim, -1, H, W)
        v_tar = self.dep_conv(v_tar.reshape(-1, self.head_dim, H, W)).reshape(B, self.num_heads, self.head_dim, -1, H, W)
        
        k = torch.stack([k_ref, k_tar], dim=1)
        v = torch.stack([v_ref, v_tar], dim=1)
        
        attn = torch.einsum('bnchw, bsnckhw -> bsnkhw', q, k).softmax(dim=-3)
        attn = self.attn_drop(attn)
        x_out = torch.einsum('bsnkhw, bsnckhw -> bsnchw', attn, v).reshape(B, 2, -1, H, W)
        
        fused_feat = torch.cat([x_out[:, 0], x_out[:, 1]], dim=1)
        
        # 内部投影: 128 -> 64
        x_fused_bottleneck = self.proj_drop(self.inner_proj(fused_feat)) 

        # ==================== 步骤 D: 空间下采样 & 通道还原 ====================
        
        # 1. 空间缩回: [B, 64, 256, 256] -> [B, 64, 32, 32]
        x_small_bottleneck = F.interpolate(x_fused_bottleneck, size=(h, w), mode='bilinear', align_corners=False)
        
        # 2. 通道还原 (Up-Projection): [B, 64, 32, 32] -> [B, 384, 32, 32]
        x_final = self.up_proj(x_small_bottleneck)
        
        # ==================== 步骤 E: 输出构造 ====================
        target_delta = x_final.flatten(2).transpose(1, 2)
        condition_zeros = torch.zeros_like(target_delta)
        
        return torch.cat([target_delta, condition_zeros], dim=1)
# 序列维度展开
class CMAAA_Transformer(nn.Module):
    # 注意：pan_channel 和 ms_channel 默认为 dim，因为 condition 是 embedding
    def __init__(self, dim, num_heads=8, pan_channel=None, ms_channel=None, 
                 pan_ks=3, ms_ks=3, ka=3, qkv_bias=False, qk_norm=False,
                 attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        
        # 如果未指定，默认 condition 通道数等于输入特征维度 dim
        self.pan_channel = dim if pan_channel is None else pan_channel
        self.ms_channel = dim if ms_channel is None else ms_channel
        
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.ka = ka

        # 生成动态卷积核的深度卷积
        self.dep_conv = nn.Conv2d(self.head_dim, self.ka * self.ka * self.head_dim, kernel_size=self.ka,
                                  bias=True, groups=self.head_dim, padding=self.ka // 2)

        # 定义 Q, K, V 卷积
        # 输入通道变为 dim + condition_dim (即 2*dim)
        self.q = nn.Conv2d(dim + self.ms_channel, dim, kernel_size=pan_ks, padding=pan_ks//2, bias=qkv_bias)
        self.k_pan = nn.Conv2d(dim + self.pan_channel, dim, kernel_size=pan_ks, padding=pan_ks//2, bias=qkv_bias)
        # 注意：原代码 v_pan 输入是 dim + pan*3 (因为有 pan, lpan, pan-lpan)，
        # 这里简化为 dim + pan_channel (只用 condition)
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
        # 初始化 shift kernel
        kernel = torch.zeros(self.ka * self.ka, self.ka, self.ka)
        for i in range(self.ka * self.ka):
            kernel[i, i // self.ka, i % self.ka] = 1.
        kernel = kernel.unsqueeze(1).repeat(self.head_dim, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, joint_features, H, W, s=None):
        """
        joint_features: (B, 2*L, D)  -> 包含拼接好的 x 和 condition
        H, W: 空间尺寸 (L = H*W)
        s: 可选的比例系数，如果不使用可移除相关逻辑
        """
        B, L_total, C = joint_features.shape
        L = H * W
        assert L_total == 2 * L + 1, "Input length must be 2 * H * W + 1"
        # 1. 拆分 x 和 condition
        x_flat, cond_flat   = joint_features.chunk(2, dim=1)
        # 调制注入
        x_flat, *_= self.norm_time(x_flat, emb=timestep.squeeze(1))

        # 2. Reshape 为 (B, C, H, W)
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        cond = cond_flat.transpose(1, 2).reshape(B, C, H, W)
        
        # 计算 Q (x + cond)
        q = self.q(torch.cat((x, cond), dim=1)).reshape(B, self.num_heads, self.head_dim, H, W)

        # 计算 K_pan (x + cond)
        k_pan = self.k_pan(torch.cat((x, cond), dim=1)).reshape(B, self.num_heads, self.head_dim, H, W)

        # 计算 V_pan (x + cond) -> 这里简化了原代码中 (x, lpan, pan, pan-lpan) 的复杂拼接
        v_pan = self.v_pan(torch.cat((x, cond), dim=1)).reshape(B, self.num_heads, self.head_dim, H, W)

        # 计算 KV_ms (x + cond)
        kv_ms = self.kv_ms(torch.cat((x, cond), dim=1)).reshape(B, 2, self.num_heads, self.head_dim, H, W)
        k_ms, v_ms = kv_ms[:, 0], kv_ms[:, 1]

        # --- 以下逻辑保持不变 ---
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
        
         # 2. 展平回序列 (B, D, H, W) -> (B, D, L) -> (B, L, D)
        cond_flat = x_pan.flatten(2).transpose(1, 2) 
        x_flat = x_ms.flatten(2).transpose(1, 2)

        # 3. 拼接回原始形状 (B, 2*L , D)
        out = torch.cat([x_flat, cond_flat], dim=1)

        return out

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.out_channels = block_kwargs["out_channels"]
        self.patch_size   = block_kwargs["patch_size"]
        self.use_CM3A     = block_kwargs["use_CM3A"]


        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.norm_ffc = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)        
        self.mlp_2 = FeedForwardControl(dim=hidden_size, dim_out=hidden_size, activation_fn="gelu-approximate")

        if self.use_CM3A:
            self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.attnCM3A = STF_CMAAA(dim=hidden_size, num_heads=num_heads, qk_norm=True, out_channels=self.out_channels, patch_size=self.patch_size)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                # nn.Linear(hidden_size, 6 * hidden_size, bias=True)
                # nn.Linear(hidden_size, 9 * hidden_size, bias=True)
                nn.Linear(hidden_size, 12 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            # nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )


    def forward(self, x, c_t, c1=None, c2=None, f1=None, c_img = None):
        if self.use_CM3A and c1 is not None:
            shift_msa, scale_msa, gate_msa, shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp, shift_mlp2, scale_mlp2, gate_mlp2 = self.adaLN_modulation(c_t).chunk(12, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_mlp2, scale_mlp2, gate_mlp2 = self.adaLN_modulation(c_t).chunk(9, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        if self.use_CM3A and c1 is not None:
            x = x + gate_attn.unsqueeze(1) * self.attnCM3A(modulate(self.norm3(x), shift_attn, scale_attn), c2, c1, f1)
        
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        # Feed Forward Control
        x = x + gate_mlp2.unsqueeze(1) * self.mlp_2(modulate(self.norm_ffc(x), shift_mlp2, scale_mlp2))

        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
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
        learn_sigma=False,
        use_CM3A = True,
        freq_CM3A = 1,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.cond_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.cond_pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
       
        if use_CM3A:
            self.blocks = nn.ModuleList([
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **{
                    "out_channels": in_channels,
                    "patch_size": patch_size,
                    "use_CM3A": i % freq_CM3A == 0,
                }) for i in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **{
                    "out_channels": in_channels,
                    "patch_size": patch_size,
                    "use_CM3A": use_CM3A,
                }) for i in range(depth)
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

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        cond_pos_embed = get_2d_sincos_pos_embed(self.cond_pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.cond_pos_embed.data.copy_(torch.from_numpy(cond_pos_embed).float().unsqueeze(0))

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
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        condition = self.cond_embedder(condition) + self.cond_pos_embed
        t = self.t_embedder(t) 

        x = torch.cat([x, condition], dim = 1)  # x 在前 condtion 在后面
        assert x.shape[1] == 2 * condition.shape[1], "x的序列长度应为 2*L"

        c_t = t
        for block in self.blocks:
            x = block(x, c_t, c1, c2, f1)    

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


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


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
    device = "cuda:2"
    model = model = DiT(
        depth=8, 
        hidden_size=384, 
        patch_size=4, 
        num_heads=12, 
        input_size=64, 
        in_channels=6, 
        learn_sigma=False
        ).to(device)
    x = torch.randn(1, 6, 64, 64).to(device)
    condition = torch.randn(1, 18, 64, 64).to(device)
    t = torch.randint(0, 1000, (x.shape[0], ), device = device)

    outputs = model(x, t, condition)
    print("successfully load DiT-S/8")
    print(f"outputs.shape is {outputs.shape}")
    print(sum(p.numel() for p in model.parameters()))
    for p in model.parameters():
        print(p.dtype)
        break
    # 输出model的大致GPU占用
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated(device)
        _ = model(x, t, condition)
        mem_bytes = torch.cuda.max_memory_allocated(device)
        mem_MB = mem_bytes / 1024**2
        print(f"Estimated GPU memory usage: {mem_MB:.2f} MB")
    except Exception as e:
        print("Unable to estimate GPU memory usage. Error:", e)
    
    # 统计前向传播的FLOPs
    try:
        from thop import profile
        flops, params = profile(model, inputs=(x, t, condition), verbose=False)
        print(f"Total FLOPs (forward): {flops}, Params: {params}")
        print(f"FLOPs per sample: {flops / x.shape[0]}")
    except Exception as e:
        print("Unable to compute FLOPs using thop. Error:", e)