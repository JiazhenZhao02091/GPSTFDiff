from functools import partial
from torch.utils.data import DataLoader
from src.data.dataloader.data_sampler import EpochBasedSampler
from src.data.dataset import SpatioTemporalFusionDataset
from src.data.transforms import LoadData, RescaleToMinusOneOne, Format
from src.metrics import RMSE, MAE, PSNR, SSIM, ERGAS, CC, SAM, UIQI

from src.model.dit.model3.diffusion_inferencer import Diffusion
from src.model.dit.model3_2.dit import DiT

dataset_cls_func = partial(
    SpatioTemporalFusionDataset,
    data_prefix_tmpl_dict=dict(
        fine_img_01="Landsat_01",
        fine_img_02="Landsat_02",
        coarse_img_01="MODIS_01",
        coarse_img_02="MODIS_02",
    ),
    data_name_tmpl_dict=dict(
        fine_img_01="{}_L_{}",
        fine_img_02="{}_L_{}",
        coarse_img_01="{}_M_{}",
        coarse_img_02="{}_M_{}",
    ),
    is_serialize_data=True,
)

transforms_key_list = [
    "fine_img_01",
    "fine_img_02",
    "coarse_img_01",
    "coarse_img_02",
]

test_transform_list = [
    LoadData(key_list=transforms_key_list),
    RescaleToMinusOneOne(key_list=transforms_key_list, data_range=[0, 10000]),
    Format(key_list=transforms_key_list),
]

test_dataset = dataset_cls_func(
    dataset_name="CIA",
    data_root="data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/test/patch",
    transform_func_list=test_transform_list,
)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    sampler=EpochBasedSampler(dataset=test_dataset, is_shuffle=False, seed=42),
    num_workers=0,
)

img_channel_num = 6


model = Diffusion(
    model_x3 = DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, input_size=64, in_channels=6, drop_path_rate=0.1, dropout=0.0),
    model_x2_x3 = DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, input_size=128, in_channels=6, drop_path_rate=0.1, dropout=0.0),
    model_x1_x2_x3 = DiT(depth=12, hidden_size=768, patch_size=16, num_heads=12, input_size=256, in_channels=6, drop_path_rate=0.1, dropout=0.0),
    model = DiT(depth=8, hidden_size=768, patch_size=4, num_heads=12, input_size=64, in_channels=6, drop_path_rate=0.0, dropout=0.0),
    image_size=256,
)

# checkpoint_path_x1_x2_x3 = "results/ditstf/syy_setting-9/model3_4/results_x1_x2_x3/checkpoints/model_epoch_4999.pth"
checkpoint_path_x1_x2_x3 = "results/ditstf/syy_setting-9/model3_2_droppath/CIA/results_x1_x2_x3/checkpoints/model_epoch_4999.pth"
checkpoint_path_x2_x3 = "results/ditstf/syy_setting-9/model3_2_droppath/CIA/results_x2_x3/checkpoints/model_epoch_4999.pth"
checkpoint_path_x3 = "results/ditstf/syy_setting-9/model3_2_droppath/CIA/results_x3/checkpoints/model_epoch_4999.pth"


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
    "test_dataloader",
    "metric_list",
    "model",
    "checkpoint_path_x3",
    "checkpoint_path_x2_x3",
    "checkpoint_path_x1_x2_x3",
]
 