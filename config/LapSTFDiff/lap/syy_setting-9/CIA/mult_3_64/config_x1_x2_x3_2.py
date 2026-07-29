from functools import partial

import torch
from torch.utils.data import DataLoader

from src.data.dataloader.data_sampler import EpochBasedSampler

from src.data.dataset import SpatioTemporalFusionDataset
from src.data.transforms import LoadData, RescaleToMinusOneOne, Format
from src.metrics import RMSE, MAE, PSNR, SSIM, ERGAS, CC, SAM, UIQI
from src.model.LapSTF.model4.diffusion import Diffusion
from src.model.LapSTFDiff import PredNoiseNetMKIRA_forward


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
]

train_dataset = dataset_cls_func(
    dataset_name='CIA',
    data_root='data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/train',
    transform_func_list=train_transform_list,
)
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=6,
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
    batch_size=4,
    num_workers=0,
    sampler=EpochBasedSampler(dataset=val_dataset, is_shuffle=False, seed=42),
)

img_channel_num = 6

model = Diffusion(
    model=PredNoiseNetMKIRA_forward(dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    image_size=256,
    T1=300,
    mode="x1+x2+x3",
)


optimizer = partial(torch.optim.Adam, lr=1e-4)

# StepLR
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


__all__ = [
    "train_dataloader",
    "val_dataloader",
    "model",
    "optimizer",
    "scheduler",
    "metric_list",
]



