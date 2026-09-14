import os
import re
from functools import partial
from pathlib import Path

from torch.utils.data import DataLoader

from src.data.dataloader.data_sampler import EpochBasedSampler
from src.data.dataset import SpatioTemporalFusionDataset
from src.data.transforms import Format, LoadData, RescaleToMinusOneOne
from src.metrics import CC, ERGAS, MAE, PSNR, RMSE, SAM, SSIM, UIQI
from src.model.GPSTFDiff import PredNoiseNetMKIRA_forward
from src.model.GPSTFDiff.diffusion_lap_single_model import Diffusion


def _epoch_from_path(path):
    match = re.search(r"epoch_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _default_checkpoint_path():
    root = Path(
        "results/GPSTFDiff/lap/Single/syy_setting-9/CIA/lap_noise_pred/results/checkpoints"
    )
    best_candidates = sorted(root.glob("best_model_epoch_*.pth"), key=_epoch_from_path)
    if best_candidates:
        return str(best_candidates[-1])
    candidates = sorted(root.glob("model_epoch_*.pth"), key=_epoch_from_path)
    if candidates:
        return str(candidates[-1])
    return str(root / "best_model_epoch_999.pth")


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
    data_root="data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/test/full",
    transform_func_list=test_transform_list,
)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    sampler=EpochBasedSampler(dataset=test_dataset, is_shuffle=False, seed=42),
    num_workers=0,
)

model = Diffusion(
    model=PredNoiseNetMKIRA_forward(
        dim=128,
        channels=6,
        out_dim=6,
        dim_mults=(1, 2, 4, 8),
        use_mkira=True,
        mkira_up_indices=(1, 2),
        mkira_init_alpha=0.0,
    ),
    image_size=256,
    sampling_timesteps=100,
    objective="pred_noise",
    noise_target="laplacian",
    T1=100,
    T2=150,
    ddim_sampling_eta=0.0,
)

checkpoint_path = os.environ.get("GPSTFDIFF_CHECKPOINT_PATH", _default_checkpoint_path())

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
    "checkpoint_path",
]
