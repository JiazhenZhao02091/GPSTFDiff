from functools import partial
import torch
from torch.utils.data import DataLoader, ConcatDataset

from src.data.dataloader.data_sampler import EpochBasedSampler
from src.data.dataloader.worker_init import worker_init_fn
from src.data.dataset import (
    SpatioTemporalFusionDatasetForSPSTFM as SpatioTemporalFusionDataset,
)
from src.data.transforms import *
from src.metrics import *
from src.model.spstfm import SPSTFM
# from src.model.ssif import SSIF


dataset_cls_func = partial(
    SpatioTemporalFusionDataset,
    data_prefix_tmpl_dict=dict(
        fine_img_01='Landsat_01',
        fine_img_02='Landsat_02',
        fine_img_03='Landsat_03',
        coarse_img_01='MODIS_01',
        coarse_img_02='MODIS_02',
        coarse_img_03='MODIS_03',
    ),
    data_name_tmpl_dict=dict(
        fine_img_01='{}_L_{}',
        fine_img_02='{}_L_{}',
        fine_img_03='{}_L_{}',
        coarse_img_01='{}_M_{}',
        coarse_img_02='{}_M_{}',
        coarse_img_03='{}_M_{}',
    ),
    is_serialize_data=True,
)

transforms_key_list = [
    'fine_img_01',
    'fine_img_02',
    'fine_img_03',
    'coarse_img_01',
    'coarse_img_02',
    'coarse_img_03',
]

train_transform_list = [
    LoadData(key_list=transforms_key_list),
    RescaleToZeroOne(key_list=transforms_key_list, data_range=[0, 10000]),
    Format(key_list=transforms_key_list),
]

train_dataset = ConcatDataset(
    [
        dataset_cls_func(
            dataset_name='CIA',
            data_root='data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/test/full',
            transform_func_list=train_transform_list,
        ),
        dataset_cls_func(
            dataset_name='LGC',
            data_root='data/spatio_temporal_fusion/LGC/private_data/syy_setting-9/test/full',
            transform_func_list=train_transform_list,
        ),
    ]
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=1,
    sampler=EpochBasedSampler(dataset=train_dataset, is_shuffle=False, seed=42),
    num_workers=0,
)


# val_transforms_list = [
#     LoadData(key_list=transforms_key_list),
#     RescaleToMinusOneOne(key_list=transforms_key_list, data_range=[0, 255]),
#     Resize(
#         key_list=['coarse_img_01', 'coarse_img_02', 'coarse_img_03'],
#         resize_shape=(64, 64),
#         interpolation_mode=0,
#         is_save_original_data=True,
#     ),
#     Format(
#         key_list=transforms_key_list
#         + ['ori_coarse_img_01', 'ori_coarse_img_02', 'ori_coarse_img_03']
#     ),
# ]

# val_dataset = ConcatDataset(
#     [
#         dataset_cls_func(
#             data_root='data/spatio_temporal_fusion/CIA/private_data/syy_setting-8/val',
#             transform_func_list=val_transforms_list,
#         ),
#         dataset_cls_func(
#             data_root='data/spatio_temporal_fusion/LGC/private_data/syy_setting-8/val',
#             transform_func_list=val_transforms_list,
#         ),
#     ]
# )

# val_dataloader = DataLoader(
#     dataset=val_dataset,
#     batch_size=1,
#     num_workers=1,
#     sampler=EpochBasedSampler(dataset=val_dataset, is_shuffle=False, seed=42),
#     worker_init_fn=partial(worker_init_fn, num_workers=1, rank=0, seed=42),
# )


patch_size = 7
sample_num = 20000
atom_num = 512
max_iter = 100
sparsity = 10
stride = 3

model = SPSTFM(
    sample_dim=patch_size**2,
    sample_num=sample_num,
    atom_num=atom_num,
    patch_size=patch_size,
    max_iter=max_iter,
    sparsity=sparsity,
)


# metric_list = [
#     RMSE(),
#     MAE(),
#     PSNR(max_value=1.0),
#     SSIM(data_range=1.0),
#     ERGAS(ratio=1.0 / 16.0),
#     CC(),
#     SAM(),
#     UIQI(data_range=1.0),
# ]

def get_model_input(self, data_per_batch):
        coarse_img_01 = data_per_batch['coarse_img_01']
        coarse_img_02 = data_per_batch['coarse_img_02'] # 如果这里用不到可以不管
        fine_img_01 = data_per_batch['fine_img_01']
        fine_img_02 = data_per_batch['fine_img_02'] # 同上

        coarse_img_03 = data_per_batch['coarse_img_03']
        fine_img_03 = data_per_batch['fine_img_03']

        # 之前计算 gradient_x 和其他内容的代码可以保留...

        # 直接让 model_input_list = [这 4 个必备参数]
        model_input_list = [coarse_img_01, coarse_img_03, fine_img_01, fine_img_03]
        
        # 兼容旧版的 input_mean 和 std，这里可以临时传 None，因为 SPSTFM 内部重新会算
        input_mean = [None, None]
        input_std = [None, None]
        
        return model_input_list, input_mean, input_std


def train_iter(
        self, iter_idx, model_input_list, input_mean, input_std, key, dataset_name
    ):
        # 此时 model_input_list 即为 [coarse_img_01, coarse_img_03, fine_img_01, fine_img_03]
        # 直接调用底层的模型多进程函数！
        (
            coarse_dictionary_matrix,
            fine_dictionary_matrix,
            sparsity_matrix,
        ) = self.model.training_dictionary_pair(*model_input_list)

        # norm_coarse_reconstruction_img = coarse_dictionary_matrix @ sparsity_matrix
        # norm_fine_reconstruction_img = fine_dictionary_matrix @ sparsity_matrix
        
        state = {
            'sparisty_matrix': sparsity_matrix,
            'coarse_dictionary_matrix': coarse_dictionary_matrix,
            'fine_dictionary_matrix': fine_dictionary_matrix,
        }
        
        save_name = key.split('-')[-1]
        save_name = save_name[:8] + '_dictionary_sparisty_' + save_name[9:]
        save_dir = self.train_dictionary_sparisty_dir / dataset_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = (
            self.train_dictionary_sparisty_dir / dataset_name / f'{save_name}.mat'
        )
        import scipy.io as scio # 确保上方有 import
        scio.savemat(save_path, state)


__all__ = [
    'train_dataloader',
    'model',
]
