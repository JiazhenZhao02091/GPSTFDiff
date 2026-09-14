import math
from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, reduce

# helpers functions
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# small helper modules

def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


# sinusoidal positional embeds


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb 's lead with learned sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        h = self.heads
        q = self.to_q(x)
        
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class SpatialCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(query_dim, inner_dim, 1, bias=False)
        self.to_k = nn.Conv2d(context_dim, inner_dim, 1, bias=False)
        self.to_v = nn.Conv2d(context_dim, inner_dim, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, query_dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        b, c, h, w = x.shape
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=self.heads), (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        
        return self.to_out(out)


class ResBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8, cross_attention_dim=None):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.conv_1 = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm_1 = nn.GroupNorm(groups, dim_out)
        self.act_1 = nn.SiLU()

        self.conv_2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm_2 = nn.GroupNorm(groups, dim_out)
        self.act_2 = nn.SiLU()
        # 确保残差连接的维度匹配
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        
        # Cross-attention layer
        self.cross_attn = None
        if exists(cross_attention_dim):
            self.cross_attn = SpatialCrossAttention(dim_out, context_dim=cross_attention_dim)

    def forward(self, x, time_emb=None, context=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.conv_1(x)
        h = self.norm_1(h)
        if exists(scale_shift):
            scale, shift = scale_shift
            h = h * (scale + 1) + shift
        h = self.act_1(h)

        h = self.conv_2(h)
        h = self.norm_2(h)
        h = self.act_2(h)

        # Apply cross-attention if context is provided
        if exists(self.cross_attn) and exists(context):
            h = h + self.cross_attn(h, context)

        return h + self.res_conv(x)

# model
class SimpleUNet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4),
        channels=6,
        resnet_block_groups=8,

        learned_sinusoidal_dim=16,
        self_condition=False,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        cross_attention_layers=None,  # 指定哪些层使用cross-attention
    ):
        super().__init__()
        # time embeddings

        time_dim = dim * 4
        # 是否使用可学习的编码        
        self.learned_sinusoidal_cond = learned_sinusoidal_cond
        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim
        # time embedding + mlp
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers

        self.channels = channels
        input_channels = self.channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 3, 1, 1)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # coarse image processing path for cross-attention
        self.coarse_init_conv = nn.Conv2d(input_channels * 3, init_dim, 3, 1, 1) # 3 Condition
        self.coarse_downs = nn.ModuleList([])

        self.noisy_init_conv = nn.Conv2d(input_channels, init_dim, 3, 1, 1)
        self.noisy_downs = nn.ModuleList([])

        self.noise_ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # Default cross-attention layers if not specified
        if cross_attention_layers is None:
            cross_attention_layers = [True] * num_resolutions  # Enable for all layers by default
        cross_attention_layers[0] = False
        print(f"cross_attention_layers: {cross_attention_layers}")

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            # Coarse image downsampling path    # DownSample中crossattentin 每次都要匹配尺度.
            coarse_down = nn.ModuleList(
                [
                    nn.Conv2d(dim_in, dim_in, 3, padding=1),
                    nn.GroupNorm(resnet_block_groups, dim_in),
                    nn.SiLU(),
                    Downsample(dim_in, dim_out)
                    if not is_last
                    else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                ]
            )
            self.coarse_downs.append(coarse_down)
            
            # Noisy image path with cross-attention
            use_cross_attn = cross_attention_layers[ind] if ind < len(cross_attention_layers) else False
            noisy_down = nn.ModuleList(
                [
                    ResBlock(
                        dim_in,
                        dim_in,
                        time_emb_dim=time_dim, # 引入Time Embedding
                        groups=resnet_block_groups,
                        cross_attention_dim=dim_in if use_cross_attn else None,
                    ),
                    Downsample(dim_in, dim_out)
                    if not is_last
                    else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                ]
            )
            self.noisy_downs.append(noisy_down)

        mid_dim = dims[-1]
        self.noisy_mid_block = ResBlock(
            mid_dim, mid_dim, time_emb_dim=time_dim, groups=8,
            cross_attention_dim=mid_dim  # Middle block also uses cross-attention
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            noise_up = nn.ModuleList(
                [
                    ResBlock(
                        dim_out + dim_in,
                        dim_out,
                        time_emb_dim=time_dim,
                        groups=resnet_block_groups,
                    ),
                    Upsample(dim_out, dim_in)
                    if not is_last
                    else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                ]
            )
            self.noise_ups.append(noise_up)

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.final_res_block = ResBlock(
            dim * 2,
            dim,
            time_emb_dim=time_dim,
            groups=resnet_block_groups,
        )
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(
        self,
        x0_t,
        time,
        condition=None,
    ):
        assert condition.shape[1] == 3 * 6, f"condition.shape[1] is {condition.shape[1]}, but should be 3 * 6"
        # Process coarse image for cross-attention context
        x_condition = self.coarse_init_conv(condition)
        print(f"x_condition shape is {x_condition.shape}")
        condition_features = []
        
        # Process noisy input
        x = self.noisy_init_conv(x0_t)
        t = self.time_mlp(time)  
        r = x.clone()

        h = []
        down_num = len(self.noisy_downs)  # 几次下采样

        # Downsampling with cross-attention
        for down_idx in range(down_num):
            # Process condition image path
            coarse_conv, coarse_norm, coarse_act, coarse_down = self.coarse_downs[down_idx]
            x_condition = coarse_conv(x_condition)
            x_condition = coarse_norm(x_condition)
            x_condition = coarse_act(x_condition)
            condition_features.append(x_condition)
            x_condition = coarse_down(x_condition)
            
            # Process noisy path with cross-attention
            res_block, downsampling = self.noisy_downs[down_idx]
            x = res_block(x, t, context=condition_features[-1])  # Apply cross-attention
            h.append(x)
            x = downsampling(x)

        # Middle block with cross-attention
        x = self.noisy_mid_block(x, t, context=x_condition)

        noise = x

        up_num = len(self.noise_ups)
        for up_idx in range(up_num):
            res_block, upsampling = self.noise_ups[up_idx]
            noise = torch.cat((noise, h.pop()), dim=1)
            noise = res_block(noise, t)
            noise = upsampling(noise)

        noise = torch.cat((noise, r), dim=1) # B C H W

        noise = self.final_res_block(noise, t)
        return self.final_conv(noise)       # return noise

if __name__ == "__main__":
    device = "cuda"
    print("Begin")
    model = SimpleUNet(
        channels=6,
        dim=64,
        dim_mults=(1, 2, 4),
        resnet_block_groups=8,
    )

    model = model.to(device)
    print("End")

    x0_t = torch.randn(1, 6, 256, 256).to(device)
    c1 = torch.randn(1, 6, 256, 256).to(device)
    c2 = torch.randn(1, 6, 256, 256).to(device)
    f1 = torch.randn(1, 6, 256, 256).to(device)
    condition = torch.cat([c1, c2, f1], dim=1)
    timesteps = torch.randint(0, 1000, (1,)).to(device)
    
    outputs = model(x0_t, timesteps, condition)

    print(f"output shape is {outputs.shape}")

    print("success!")
