import torch
import timm
import torchsummary
import torch.nn as nn
from typing import Optional, List
import copy
import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.nn.functional as F
from torch import nn, Tensor
import os
from torchvision.transforms import Normalize
import math

def _get_backbone_dino():
    backbone = timm.create_model(
        "vit_large_patch16_dinov3.sat493m",
        pretrained=False,
        num_classes=0,
    )
    # load model weights
    model_path="/home/zhaojiazhen/workspace/STF/test/dinov3_model.pth"
    checkpoint = torch.load(model_path, map_location='cpu')
    backbone.load_state_dict(checkpoint)
    #TODO: 修改模型的输入通道


    return backbone

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=6, d_model=512, patch_size=16, image_size=256):
        super().__init__()
        self.pathEmbedding = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.H_patchs = self.W_patchs = image_size / patch_size # 16

    def forward(self, x):
        # print(x.shape)      # 1 6 256 256
        x = self.pathEmbedding(x)
        # print(x.shape)      # 1 1024 16 16
        return x
        
class LinearProjection(nn.Module):
    def __init__(self, in_channels=512, d_model=1024):
        super().__init__()
        self.linearProjection = nn.Linear(in_channels, d_model)

    def forward(self, x):
        x = x.flatten(2) # B C HW
        x = x.permute(0, 2, 1)  # B N/HW C
        x = self.linearProjection(x)
        return x

class TransformerEncoder(nn.Module):
    """
        Layers + norm
    """
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.embed=PatchEmbedding()
        self.linear_proj=LinearProjection()

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        # TODO PatchEmbedding  + 位置编码
        # print(f"output shape is {output.shape}")
        output=self.embed(output)
        output=self.linear_proj(output)
        # print(f"output shape is {output.shape}")
        # TODO 位置编码
        # 为output添加固定的正弦位置编码
        B, N, _ = output.shape
        pe = self._get_sinusoidal_pos_embed(N, output.size(-1), device=output.device, dtype=output.dtype)
        output = output + pe.unsqueeze(0)

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
        # TODO:???
        # if self.norm is not None:
        #     output = self.norm(output)

        return output
    def _get_sinusoidal_pos_embed(self, num_pos, d_model, device=None, dtype=None):
        """生成固定的正弦位置编码"""
        pe = torch.zeros(num_pos, d_model, device=device, dtype=dtype)
        position = torch.arange(0, num_pos, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos  # 加入位置编码
    # 自注意力机制 + 前馈神经网络
    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)   # 引入位置编码

        # MultiheadAttention
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        
        # print("======================== Decoder =========================")
        # print(f"output shape is {output.shape}")
        # print(f"memory shape is {memory.shape}")

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        # TODO:???
        # if self.norm is not None:
        #     output = self.norm(output)
        #     if self.return_intermediate:
        #         intermediate.pop()
        #         intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        # print(f"output shape {output.shape}")
        # print(f"output.unsqueeze(0) shape {output.unsqueeze(0).shape}")
        # return output.unsqueeze(0)
        return output

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # d_model embedding dim
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        # if pos is None:
            # print(f"pos is None")

        q = self.with_pos_embed(tgt, query_pos) # fine image
        k = self.with_pos_embed(memory, pos)    # extra feature
        v = memory                              # extra feature
 
        tgt2 = self.self_attn(q, k, v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
    
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class UpsampleDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 16 16
        self.decoder = nn.Sequential(
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(512, 256, (3, 3)),
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2, mode='nearest'),  # 32
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(256, 256, (3, 3)),
                        nn.ReLU(),
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(256, 256, (3, 3)),
                        nn.ReLU(),
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(256, 256, (3, 3)),
                        nn.ReLU(),
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(256, 128, (3, 3)),
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2, mode='nearest'), # 64
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(128, 128, (3, 3)),
                        nn.ReLU(),
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(128, 64, (3, 3)),
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2, mode='nearest'), # 128
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(64, 64, (3, 3)),
                        nn.ReLU(),
                        nn.ReflectionPad2d((1, 1, 1, 1)),
                        nn.Conv2d(64, 6, (3, 3)),
                        nn.Upsample(scale_factor=2, mode='nearest'), # 256
                    )
    def forward(self, x):
        return self.decoder(x)


class DinoFrozen(nn.Module):
    def __init__(self, backbone=None, d_model=1024, n_heads=8, 
                 num_encoder_layers=3, num_decoder_layers=3, encoder_norm=False):
        super().__init__()
        # self.backbone = backbone
        self.backbone = _get_backbone_dino()
        # freeze backbone
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.linear1 = nn.Linear(d_model*2, d_model)
        # Encoder
        self.encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=n_heads)
        self.encoder = TransformerEncoder(self.encoder_layer, num_encoder_layers, encoder_norm)
        
        # Decoder
        self.decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=n_heads)
        self.decoder = TransformerDecoder(self.decoder_layer, num_decoder_layers, encoder_norm) # 1 256 1024

        # Upsample
        self.linear = nn.Linear(d_model, 512)
        self.upsample = UpsampleDecoder()

        # normlize
        self.n = Normalize(mean=torch.tensor([0.4300, 0.4110, 0.2960]), std=torch.tensor([0.2130, 0.1560, 0.1430]))
    def norm(x):
        x = self.n(x)
        return x
    def forward(self, coarse_1, coarse_2, fine_1, fine_2):
        # 6 256 256
        time_diff = coarse_2 - coarse_1
        sensor_diff = fine_1 - coarse_1
        def _get_mean_fea(x):
            x_pre = x[:, :3, :, :]
            x_post= x[:, 3:, :, :]
            # 需要使用官方进行初始化
            # x_pre_fea = self.backbone.forward_features(self.n(x_pre))
            # x_post_fea = self.backbone.forward_features(self.n(x_post))
            # 未初始化
            x_pre_fea = self.backbone.forward_features(x_pre)
            x_post_fea = self.backbone.forward_features(x_post)
            # B 261 D
            mean_fea = (x_pre_fea + x_post_fea) / 2.0
            # print(f"mean_fea shape {mean_fea.shape}")
            mean_fea = mean_fea[:, 5:, :]
            return mean_fea

        # time_fea = self.backbone.forward_features(time_diff)
        # sensor_fea = self.backbone.forward_features(sensor_diff)
        time_fea = _get_mean_fea(time_diff)         # 1 256 1024
        sensor_fea = _get_mean_fea(sensor_diff)     # 1 256 1024
        
        # print(f"time_fea shape:{time_fea.shape}, sensor_fea shape: {sensor_fea.shape}")
        fea_diff = torch.cat((time_fea, sensor_fea), dim=2) # 1 256 2048
        # print(f"fea shape: {fea_diff.shape}")   # B N C
        fea_diff = self.linear1(fea_diff) # 1 256 1024
        # print(f"fea shape: {fea_diff.shape}")

        fine_1_fea = self.encoder(fine_1)   # 1 256 1024
        B, N, D = fine_1_fea.shape
        # print(f"B: {B}, N: {N}, D: {D}")

        out_decoded = self.decoder(fine_1_fea, fea_diff)        # Q  KV

        # print(f"fine_1_fea shape: {fine_1_fea.shape}, out_decoded shape: {out_decoded.shape}")

        out_decoded = self.linear(out_decoded)
        # print(f"out_decoded shape: {out_decoded.shape}")

        output = out_decoded.permute(0, 2, 1) # B C N --> B C H W
        H = W = int(math.sqrt(N)) # 16
        output = output.reshape(B, 512, H, W)     # B 6 16 16


        return self.upsample(output)


if __name__ == "__main__":

    coarse1 = torch.randn(1, 6, 256, 256)
    coarse2 = torch.randn(1, 6, 256, 256)
    fine1 = torch.randn(1, 6, 256, 256)
    fine2 = torch.randn(1, 6, 256, 256)
    model = DinoFrozen()
    output = model(coarse1, coarse2, fine1, fine2)  
    print(f"output shape: {output.shape}")


    # backbone = _get_backbone_dino()
    # device = "cuda:2"
    # # x = torch.randn(1, 6, 256, 256)
    # # x = torch.randn(1, 3, 256, 256)
    # x = torch.randn(1, 3, 512, 512)
    # x = x.to(device)
    # backbone = backbone.to(device)

    # # get model specific transforms (normalization, resize)
    # data_config = timm.data.resolve_model_data_config(backbone)
    # train_transforms = timm.data.create_transform(**data_config, is_training=True)
    # val_transforms = timm.data.create_transform(**data_config, is_training=False)

    # print(data_config)
    # print(train_transforms)
    # print(val_transforms)
    # # TOOD 只需要Resize + 转Tensor float 以及 Normalization

    # out = backbone.forward_features(x) # x : 1 261 1024
    # print(out.shape)
    # out = out[:, 5: , :]
    # print(out.shape)

    # backbone = _get_backbone_dino()
    # device = "cuda:2"
    # x = torch.randn(1, 6, 256, 256)

    # # transform = torchvision.transforms.Compose([
    # #     torchvision.transforms.ToTensor(),
    # #     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # # ])

    # # x = transform(x)

    # print(x.shape)

    # # 按通道维度平均切分为两个Tensor
    # x1, x2 = torch.chunk(x, 2, dim=1)  # 在通道维切成两份，每份3通道

    # print(f"x1 shape {x1.shape}, x2 shape{x2.shape}")