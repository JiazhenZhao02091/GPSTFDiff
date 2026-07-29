import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class SPCSA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(SPCSA, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.linear_0 = nn.Conv2d(dim, dim , 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)                                                                                                     # 微信公众号:AI缝合术
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, 1, kernel_size=1),  # 输出动态 K
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.linear_0(x)

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)                                                                                                     # 微信公众号:AI缝合术
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)                                                                                                     # 微信公众号:AI缝合术
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)                                                                                                     # 微信公众号:AI缝合术

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape
        dynamic_k = int(C * self.gate(x).view(b, -1).mean())
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        mask = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)                                                                                                     # 微信公众号:AI缝合术
        index = torch.topk(attn, k=dynamic_k, dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.)
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))

        attn = attn.softmax(dim=-1)
        out1 = (attn @ v)
        out2 = (attn @ v)
        out3 = (attn @ v)
        out4 = (attn @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4                                                                                                     # 微信公众号:AI缝合术

        print(f"out shape is{out.shape}")
        out_att = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)                                                                                                     # 微信公众号:AI缝合术
        print(f"out_att shape is{out_att.shape}")
        return out_att
    
    
if __name__ == "__main__":

    # 输入张量：形状为 (B, C, H, W)
    x = torch.randn(1, 32, 64, 64)

    # 初始化 SPCSA 模块
    spcsa = SPCSA(dim=32, num_heads=4, bias=True)

    # 前向传播测试
    output = spcsa(x)

    # 输出结果形状
    print(spcsa)
    print("\n微信公众号:AI缝合术\n")
    print("输入张量形状:", x.shape)       # [B, C, H, W]
    print("输出张量形状:", output.shape)  # [B, C, H, W]
