from torch.nn.modules.module import Module
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
from timm.layers import to_2tuple


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
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
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
    """
    创建一个注意力掩码 (N, N)
    0 表示可见，-inf 表示不可见
    """
    N = N_t + N_c + N_x
    # 初始化全为 0 (默认全可见)
    mask = torch.zeros((N, N))
    # 定义各部分的索引范围
    start_c = N_t
    end_c = N_t + N_c
    start_x = end_c
    # --- 设置 Condition 的可见性 ---
    # Condition (Rows: start_c ~ end_c)
    # 需求：Condition 只关注自身
    # 1. 禁止 Condition 看 Timestep (0 ~ start_c)
    if start_c > 0:
        mask[start_c:end_c, :start_c] = float('-inf')
        
    # 2. 禁止 Condition 看 X (start_x ~ end)
    # 这里的 X 是含噪影像，通常防止信息泄露给 Condition
    mask[start_c:end_c, start_x:] = float('-inf')
    
    # --- 设置 X 的可见性 ---
    # 需求：X 看全局 -> 不需要修改，保持为 0 即可
    
    return mask

def get_freq_indices(method):
    """
    复原 M2SFormer 的频率索引生成逻辑
    """
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

''' Attention Block '''
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
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

''' Local Conv MLP ''' 
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
                timestep, patch_tokens = hidden_states[:, :1, :], hidden_states[:, 1:, :]
                hidden_states_control_org, hidden_states  = patch_tokens.chunk(2, dim=1)
                B, N, C = hidden_states.shape
                h = w = int(np.sqrt(N))
                assert h * w == N
                hidden_states_control = hidden_states_control_org.reshape(B, h, w, C).permute(0, 3, 1, 2)
                hidden_states_control = self.control_conv(hidden_states_control)
                hidden_states_control = hidden_states_control.reshape(B, C, N).permute(0, 2, 1)
                hidden_states = hidden_states + 1.2 * hidden_states_control # TODO: add control signal, better change to 1.0 when training
                hidden_states = torch.cat([timestep, hidden_states_control_org, hidden_states], dim=1)
        # print(f"hidden_states shape is {hidden_states.shape}, timestep shape is {timestep.shape}")
        # hidden_states = hidden_states = torch.cat([timestep, hidden_states], dim=1)
        return hidden_states

''' Cross-Modality Alignment-Aware Attention (CM3A) '''
# Encoder部分可以考虑不使用CMAA
class CMAAA(nn.Module):
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
        timestep, patch_tokens = joint_features[:, :1, :], joint_features[:, 1:, :]
        cond_flat, x_flat  = patch_tokens.chunk(2, dim=1)
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

        # 3. 拼接回原始形状 (B, 2*L+1 , D)
        out = torch.cat([timestep, cond_flat, x_flat], dim=1)

        return out

""" MS2 Module"""
class MSMSCore(nn.Module):
    """
    这是从 M2SFormer 中剥离出的核心算法。
    为了适配 U-ViT，我们去掉了原始代码中“处理 List 输入并拼接”的部分，
    因为 U-ViT 单层只有一个特征图。但内部的 DCT 和 多尺度卷积逻辑完全保留。
    """
    def __init__(self,
                 in_channels: int,
                 target_resolution: int = 64, # DCT 计算的基准分辨率
                 scale_branches: int = 2,     # 多尺度分支数
                 min_channel: int = 64,       # 分支最小通道
                 min_resolution: int = 8,
                 frequency_branches: int = 16, # DCT 频率分量数
                 frequency_selection: str = 'top',
                 reduction: int = 16) -> None:
        super().__init__()

        # --- 配置检查 ---
        assert frequency_branches in [1, 2, 4, 8, 16, 32]
        assert frequency_selection in ['top', 'bottom', 'low']
        frequency_selection = frequency_selection + str(frequency_branches)

        self.scale_branches = scale_branches
        self.min_channel = min_channel
        self.min_resolution = min_resolution
        self.target_resolution = to_2tuple(target_resolution)
        self.num_freq = frequency_branches
        self.dct_h, self.dct_w = self.target_resolution

        # ==========================================
        # Part A: Multi-Spectral Attention (DCT)
        # ==========================================
        mapper_x, mapper_y = get_freq_indices(frequency_selection)
        # 映射到当前 DCT 分辨率
        mapper_x = [temp_x * self.dct_h // 7 for temp_x in mapper_x]
        mapper_y = [temp_y * self.dct_w // 7 for temp_y in mapper_y]

        # 注册固定的 DCT 滤波器权重
        for freq_idx in range(frequency_branches):
            self.register_buffer('dct_weight_{}'.format(freq_idx), 
                                 self.get_dct_filter(self.dct_h, self.dct_w,
                                                     mapper_x[freq_idx], mapper_y[freq_idx],
                                                     in_channels))

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False))

        self.avg_channel_pool = nn.AdaptiveAvgPool2d(1)
        self.max_channel_pool = nn.AdaptiveMaxPool2d(1)

        # ==========================================
        # Part B: Multi-Scale Spatial Attention
        # ==========================================
        for scale_idx in range(scale_branches):
            inter_channels = in_channels // 2 ** scale_idx
            if inter_channels < self.min_channel: inter_channels = self.min_channel

            # 关键：Dilation 随着 scale 增加 (1+scale_idx)
            init_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, 
                          padding=(1 + scale_idx), dilation=(1 + scale_idx), bias=False),
                nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True),
            )

            spatial_attention_map = nn.Sequential(
                nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
                nn.Sigmoid()
            )

            final_conv = nn.Sequential(
                nn.Conv2d(inter_channels, in_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            )

            setattr(self, 'init_conv_scale_{}'.format(scale_idx), init_conv)
            setattr(self, 'spatial_attention_map_scale_{}'.format(scale_idx), spatial_attention_map)
            setattr(self, 'final_conv_scale_{}'.format(scale_idx), final_conv)

        # 可学习的加权参数 Alpha / Beta
        self.alpha_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(scale_branches)])
        self.beta_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(scale_branches)])

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, in_channels):
        dct_filter = torch.zeros(in_channels, tile_size_x, tile_size_y)
        for t_x in range(tile_size_x):
            for t_y in range(tile_size_y):
                dct_filter[:, t_x, t_y] = self.build_filter(t_x, mapper_x, tile_size_x) * \
                                          self.build_filter(t_y, mapper_y, tile_size_y)
        return dct_filter

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.size()

        # --- STEP 1: Multi-Spectral Attention (频域处理) ---
        x_pooled = x
        # 强制对齐到 DCT 分辨率 (例如 64x64)
        if H != self.dct_h or W != self.dct_w:
            x_pooled = F.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

        multi_spectral_feature_avg, multi_spectral_feature_max = 0, 0
        for freq_idx in range(self.num_freq):
            # 获取注册的 DCT 权重
            dct_weight = getattr(self, 'dct_weight_{}'.format(freq_idx))
            x_pooled_spectral = x_pooled * dct_weight
            multi_spectral_feature_avg += self.avg_channel_pool(x_pooled_spectral)
            multi_spectral_feature_max += self.max_channel_pool(x_pooled_spectral)

        multi_spectral_feature_avg /= self.num_freq
        multi_spectral_feature_max /= self.num_freq

        # MLP + Sigmoid
        multi_spectral_feature_avg = self.fc(multi_spectral_feature_avg).view(B, C, 1, 1)
        multi_spectral_feature_max = self.fc(multi_spectral_feature_max).view(B, C, 1, 1)
        multi_spectral_attention_map = torch.sigmoid(multi_spectral_feature_avg + multi_spectral_feature_max)

        # 增强后的特征
        cross_feature_map = x * multi_spectral_attention_map.expand_as(x)

        # --- STEP 2: Multi-Scale Attention (空洞卷积) ---
        refine_feature_map = 0
        for scale_idx in range(self.scale_branches):
            init_conv = getattr(self, 'init_conv_scale_{}'.format(scale_idx))
            spatial_attn_op = getattr(self, 'spatial_attention_map_scale_{}'.format(scale_idx))
            final_conv = getattr(self, 'final_conv_scale_{}'.format(scale_idx))

            # 下采样以节省计算 (如果特征图足够大)
            if int(cross_feature_map.shape[2] // 2 ** scale_idx) >= self.min_resolution:
                feature_scale = F.avg_pool2d(cross_feature_map, kernel_size=2 ** scale_idx, stride=2 ** scale_idx)
            else:
                feature_scale = cross_feature_map

            # 卷积分支
            feat = init_conv(feature_scale)
            attn = spatial_attn_op(feat)
            
            # Alpha / Beta 加权融合
            feat = feat * attn * self.alpha_list[scale_idx] + feat * (1 - attn) * self.beta_list[scale_idx]
            feat = final_conv(feat)

            # 上采样回原分辨率
            if feat.shape[2:] != (H, W):
                feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=True)
            
            refine_feature_map += feat

        refine_feature_map = refine_feature_map / self.scale_branches
        # 残差连接: 原始输入 + 频域增强 + 多尺度增强
        return x + refine_feature_map # 这里的 x 可以是 x_out (频域增强后) 或 原始 x，原论文逻辑是累加

class UViT_MSMS_Adapter(nn.Module):
    def __init__(self, dim, target_resolution=64, scale_branches=2, frequency_branches=16):
        super().__init__()
        # 初始化核心模块
        self.msms_core = MSMSCore(in_channels=dim,
                                  target_resolution=target_resolution,
                                  scale_branches=scale_branches,
                                  frequency_branches=frequency_branches)
        
        # 可选：层归一化，保证 Token 进出分布一致
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W, t_len=1, cond_len=0):
        """
        x: [B, L, C] - 拼接后的 Tokens (Time + Condition + Image)
        H, W: 图像部分 Token 对应的空间尺寸 (H*W = Image Tokens)
        t_len: Timestep Token 的长度 (通常为 1)
        cond_len: Condition Tokens 的长度
        """
        B, L, C = x.shape
        
        # 1. 拆分序列
        # 假设顺序是 [Time, Condition, Image]
        idx_img_start = t_len + cond_len
        
        tokens_others = x[:, :idx_img_start, :] # Time + Condition
        tokens_img = x[:, idx_img_start:, :]    # Image
        
        assert tokens_img.shape[1] == H * W, f"Image tokens length {tokens_img.shape[1]} != H*W ({H}*{W})"

        # 2. Reshape Image Tokens -> 2D Feature Map
        # [B, HW, C] -> [B, C, H, W]
        feat_img = tokens_img.transpose(1, 2).view(B, C, H, W)
        
        # 3. 送入 MSMS 核心模块增强
        # 这里进行 DCT 频域处理和多尺度空洞卷积
        feat_img_enhanced = self.msms_core(feat_img)
        
        # 4. Flatten Back -> Tokens
        # [B, C, H, W] -> [B, HW, C]
        tokens_img_enhanced = feat_img_enhanced.flatten(2).transpose(1, 2)
        # 5. 拼接还原
        # 我们只增强了图像部分，Condition 和 Timestep 保持原样
        x_out = torch.cat([tokens_others, tokens_img_enhanced], dim=1)
        
        # 残差连接通常由外部 Block 处理，这里直接返回处理后的全序列
        return self.norm(x_out)

# U-ViT Block
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False, grid_size=None, use_CMAAA=True, use_local_mlp=True):
        super().__init__()
       
        
        self.grid_size = grid_size
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)  # 目前只考虑了自注意力机制，即条件和含噪影像都会被使用
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)

        if use_local_mlp:
            self.mlp = FeedForwardControl(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        else:
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer) # hidden_features is the dimension of the output of the MLP
        
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
        if self.skip_linear is not None: # Decoder default skip is True
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x), attn_mask=mask)
        if self.use_CMAAA:
            x = x + self.CM3A(joint_features = self.norm3(x), H=self.grid_size, W=self.grid_size)
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
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
                 use_checkpoint=False, conv=True, skip=True, has_img=True, use_mask=False, use_CMAAA=True, use_local_mlp=True, use_skip_adapter=True):
        super().__init__()
        
        print(f"use_CMAAA is {use_CMAAA}, use_loacl_mlp is {use_local_mlp}, use_skip_adapter is {use_skip_adapter}")
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.has_img = has_img

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.cond_init_conv = nn.Conv2d(in_chans * 3, in_chans, 3, padding=1)
        self.cond_embed  = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = (img_size // patch_size) ** 2

        self.grid_size = img_size // patch_size
        self.use_skip_adapter = use_skip_adapter
        self.use_CMAAA = use_CMAAA
        self.use_local_mlp = use_local_mlp
        
        if use_mask:
            self.register_buffer('mask', create_custom_mask(N_t = 1, N_c  = num_patches, N_x = num_patches))
        else:
            self.mask = None
        
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()
        
        # the number of extra tokens 
        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, embed_dim)
            self.extras = 2
        else:
            self.extras = 1

        if self.has_img:
            self.extras += num_patches  # cond_embed
        # 条件图像和噪声图像一起添加位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, embed_dim)) # B 2*L+1 D
        
        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint, grid_size = self.grid_size, use_CMAAA=use_CMAAA, use_local_mlp=use_local_mlp)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint, grid_size = self.grid_size, use_CMAAA=use_CMAAA, use_local_mlp=use_local_mlp)
        
        
        if self.use_skip_adapter:
            self.skip_adapters = nn.ModuleList([
                UViT_MSMS_Adapter(
                    dim=embed_dim, 
                    target_resolution=self.grid_size, 
                    scale_branches=2, 
                    frequency_branches=8
                ) for _ in range(depth // 2) 
            ])

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
        H = W = self.grid_size

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)
        # label embedding
        if y is not None:
            cond_img = self.cond_init_conv(y)
            cond_img = self.cond_embed(cond_img)
            x = torch.cat((cond_img, x), dim=1)  # B 2*L D

        x = torch.cat((time_token, x), dim=1) # timestep cond_img x
        x = x + self.pos_embed

        skips = []
        for blk in self.in_blocks:
            x = blk(x, mask=self.mask)
            skips.append(x)

        x = self.mid_block(x, mask=self.mask)
        
        if self.use_skip_adapter:
            for i, blk in enumerate(self.out_blocks):
                skip = skips.pop()
                skip_refined = self.skip_adapters[i](
                    skip, 
                    H=H, W=W, 
                    t_len=1, 
                    cond_len=H*W # ??
                )
                x = blk(x, skip=skip_refined, mask=self.mask)
        else:
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

    """
        * 低频用小的patch size
        * 高频用大的patch size
        * 因为高频信息中很多都是无意义的像素点即高频信息比较稀疏
    """
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
    x = torch.randn(1, 3, 256, 256).to(device)
    c1 = torch.randn(1, 3, 256, 256).to(device)
    c2 = torch.randn(1, 3, 256, 256).to(device)
    f1 = torch.randn(1, 3, 256, 256).to(device)
    condition = torch.cat([c1, c2, f1], dim=1)
    timesteps = torch.randint(0, 1000, (1,)).to(device)
    out = model(x, timesteps, condition)
    print(out.shape)