from functools import partial
import sched

import torch
from torch.utils.data import DataLoader

from src.data.dataloader.data_sampler import EpochBasedSampler

from src.data.dataset import SpatioTemporalFusionDataset
from src.data.transforms import *
from src.metrics import *
from torchvision.transforms import ToTensor, Normalize, Compose
from src.model.UViT.model1.UViT import UViT



dataset_cls_func = partial(
    SpatioTemporalFusionDataset,
    data_prefix_tmpl_dict=dict(
        fine_img_01='Landsat_01',
        fine_img_02='Landsat_02',
        coarse_img_01='MODIS_01',
        coarse_img_02='MODIS_02',
    ),
    data_name_tmpl_dict=dict(
        fine_img_01='{}_L_{}',
        fine_img_02='{}_L_{}',
        coarse_img_01='{}_M_{}',
        coarse_img_02='{}_M_{}',
    ),
    is_serialize_data=True,
)

transforms_key_list = [
    'fine_img_01',
    'fine_img_02',
    'coarse_img_01',
    'coarse_img_02',
]

train_transform_list = [
    LoadData(key_list=transforms_key_list),
    RescaleToMinusOneOne(key_list=transforms_key_list, data_range=[0, 10000]),
    Format(key_list=transforms_key_list),
    # Normalize(mean=torch.tensor([0.4300, 0.4110, 0.2960]), std=torch.tensor([0.2130, 0.1560, 0.1430])),
]

train_dataset = dataset_cls_func(
    dataset_name='LGC',
    data_root='data/spatio_temporal_fusion/LGC/private_data/syy_setting-9/train',
    transform_func_list=train_transform_list,
)
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    sampler=EpochBasedSampler(dataset=train_dataset, is_shuffle=True, seed=42),
    num_workers=4,
)


val_transforms_list = [
    LoadData(key_list=transforms_key_list),
    RescaleToMinusOneOne(key_list=transforms_key_list, data_range=[0, 10000]),
    Format(key_list=transforms_key_list),
    # Normalize(mean=torch.tensor([0.4300, 0.4110, 0.2960]), std=torch.tensor([0.2130, 0.1560, 0.1430])),
]

val_dataset = dataset_cls_func(
    dataset_name='LGC',
    data_root='data/spatio_temporal_fusion/LGC/private_data/syy_setting-9/val',
    transform_func_list=val_transforms_list,
)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=16,
    num_workers=0,
    sampler=EpochBasedSampler(dataset=val_dataset, is_shuffle=False, seed=42),
)

img_channel_num = 6

model = UViT(img_size=256, patch_size=16, in_chans=6, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, mlp_time_embed=False,
                 use_checkpoint=False, conv=True, skip=True, has_image=True)


optimizer = partial(torch.optim.Adam, lr=1e-4)

# scheduler = partial(torch.optim.lr_scheduler.ExponentialLR, optimizer=optimizer, gamma=0.99)

scheduler = None

metric_list = [
    RMSE(),
    MAE(),
    PSNR(max_value=1.0),
    SSIM(data_range=1.0),
    ERGAS(ratio=1.0 / 16.0),
    CC(),
    SAM(),
    UIQI(),
]

# loss_fun = 'mse'
loss_fun = 'l1'


__all__ = [
    'train_dataloader',
    'val_dataloader',
    'model',
    'optimizer',
    'scheduler',
    'metric_list',
    'loss_fun',
]
