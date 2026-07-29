from functools import partial
import torch
from torch.utils.data import DataLoader
from src.data.dataloader.data_sampler import EpochBasedSampler
from src.data.dataset import SpatioTemporalFusionDataset
from src.data.transforms import LoadData, RescaleToMinusOneOne, Format, Rotate, Flip
from src.metrics import RMSE, MAE, PSNR, SSIM, ERGAS, CC, SAM, UIQI

from src.model.dit.model3_5.diffusion import Diffusion
from src.model.dit.model3_5.dit import DiT

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
    Rotate(key_list=transforms_key_list),
    Flip(key_list=transforms_key_list),
    Format(key_list=transforms_key_list),
]

train_dataset = dataset_cls_func(
    dataset_name='CIA',
    data_root='data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/train',
    transform_func_list=train_transform_list,
)
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=24,
    sampler=EpochBasedSampler(dataset=train_dataset, is_shuffle=True, seed=42),
    num_workers=4,
)


val_transforms_list = [
    LoadData(key_list=transforms_key_list),
    RescaleToMinusOneOne(key_list=transforms_key_list, data_range=[0, 10000]),
    Format(key_list=transforms_key_list),
]

val_dataset = dataset_cls_func(
    dataset_name='CIA',
    data_root='data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/val',
    transform_func_list=val_transforms_list,
)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=16,
    num_workers=0,
    sampler=EpochBasedSampler(dataset=val_dataset, is_shuffle=False, seed=42),
)

img_channel_num = 6

model = Diffusion(
    model = DiT(
        depth=8, 
        hidden_size=768, 
        patch_size=4, 
        num_heads=12, 
        input_size=64, 
        in_channels=6,
        drop_path_rate=0.0,
        dropout=0.0,
    ),
    image_size=256,
    mode="x3",
)


optimizer = partial(torch.optim.Adam, lr=1e-4)

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


__all__ = [
    "train_dataloader",
    "val_dataloader",
    "model",
    "optimizer",
    "scheduler",
    "metric_list",
]



