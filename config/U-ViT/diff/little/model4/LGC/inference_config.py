from functools import partial
from torch.utils.data import DataLoader
from src.data.dataloader.data_sampler import EpochBasedSampler
from src.data.dataset import SpatioTemporalFusionDataset
from src.data.transforms import LoadData, RescaleToMinusOneOne, Format
from src.metrics import RMSE, MAE, PSNR, SSIM, ERGAS, CC, SAM, UIQI
from src.model.LapSTF.model4.pred_resnet import PredNoiseNet
from src.model.LapSTF.model4.diffusion_inferencer import Diffusion
# src/model/LapSTF/model4/diffusion.py
# src/inferencer/lapstf/model4/lapSTF_inferencer.py
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
    data_root="data/spatio_temporal_fusion/LGC/private_data/syy_setting-9/test/full",
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
    model_x3 = PredNoiseNet(dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4)),
    model_x2_x3 = PredNoiseNet(dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4)),
    model_x1_x2_x3 = PredNoiseNet(dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4)),
    model = PredNoiseNet(dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4)),
    image_size=256,
)

checkpoint_path_x1_x2_x3 = "/home/zhaojiazhen/workspace/STF/STF/results/LapSTF/model4/CIA/results_x1_x2_x3/checkpoints/model_epoch_4999.pth"
checkpoint_path_x2_x3 = "/home/zhaojiazhen/workspace/STF/STF/results/LapSTF/model4/CIA/results_x2_x3/checkpoints/model_epoch_4999.pth"
checkpoint_path_x3 = "/home/zhaojiazhen/workspace/STF/STF/results/LapSTF/model4/CIA/results_x3/checkpoints/model_epoch_4999.pth"

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
