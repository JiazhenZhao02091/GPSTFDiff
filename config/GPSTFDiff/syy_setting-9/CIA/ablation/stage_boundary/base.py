from functools import partial
from pathlib import Path

from torch.utils.data import DataLoader

from src.data.dataloader.data_sampler import EpochBasedSampler
from src.data.dataset import SpatioTemporalFusionDataset
from src.data.transforms import Format, LoadData, RescaleToMinusOneOne
from src.metrics import CC, ERGAS, MAE, PSNR, RMSE, SAM, SSIM, UIQI
from src.model.GPSTFDiff import PredNoiseNetMKIRA
from src.model.GPSTFDiff.diffusion_lap_inferencer import Diffusion


DATA_ROOT = "data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/test/full"
CHECKPOINT_PATH_X3 = "/home/zhaojiazhen/workspace/STF/STF/checkpoints/best_model_epoch_149.pth"
CHECKPOINT_PATH_X2_X3 = "/home/zhaojiazhen/workspace/STF/STF/checkpoints/best_model_epoch_149.pth"
CHECKPOINT_PATH_X1_X2_X3 = "/home/zhaojiazhen/workspace/STF/STF/checkpoints/best_model_epoch_149.pth"

DEFAULT_NUM_TRAIN_TIMESTEPS = 1000
DEFAULT_SAMPLING_TIMESTEPS = 100
DEFAULT_T1 = 100
DEFAULT_T2 = 300


def _path_from_env(name, default):
    """Let the batch runner override machine-specific paths without editing configs."""
    import os

    return os.environ.get(name, default)


def build_config(
    *,
    t1,
    t2,
    sampling_timesteps=DEFAULT_SAMPLING_TIMESTEPS,
    num_train_timesteps=DEFAULT_NUM_TRAIN_TIMESTEPS,
    data_root=DATA_ROOT,
    checkpoint_path_x3=CHECKPOINT_PATH_X3,
    checkpoint_path_x2_x3=CHECKPOINT_PATH_X2_X3,
    checkpoint_path_x1_x2_x3=CHECKPOINT_PATH_X1_X2_X3,
    batch_size=1,
    num_workers=0,
):
    data_root = _path_from_env("GPSTFDIFF_STAGE_DATA_ROOT", data_root)
    checkpoint_path_x3 = _path_from_env(
        "GPSTFDIFF_STAGE_CKPT_X3", checkpoint_path_x3
    )
    checkpoint_path_x2_x3 = _path_from_env(
        "GPSTFDIFF_STAGE_CKPT_X2_X3", checkpoint_path_x2_x3
    )
    checkpoint_path_x1_x2_x3 = _path_from_env(
        "GPSTFDIFF_STAGE_CKPT_X1_X2_X3", checkpoint_path_x1_x2_x3
    )

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
        data_root=data_root,
        transform_func_list=test_transform_list,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        sampler=EpochBasedSampler(dataset=test_dataset, is_shuffle=False, seed=42),
        num_workers=num_workers,
    )

    model_kwargs = dict(
        dim=128,
        channels=6,
        out_dim=6,
        dim_mults=(1, 2, 4, 8),
        use_mkira=True,
        mkira_up_indices=(1, 2),
        mkira_init_alpha=0.0,
    )

    model = Diffusion(
        model_x3=PredNoiseNetMKIRA(**model_kwargs),
        model_x2_x3=PredNoiseNetMKIRA(**model_kwargs),
        model_x1_x2_x3=PredNoiseNetMKIRA(**model_kwargs),
        model=PredNoiseNetMKIRA(
            dim=64,
            channels=6,
            out_dim=6,
            dim_mults=(1, 2, 4),
            use_mkira=True,
            mkira_up_indices=(1, 2),
            mkira_init_alpha=0.0,
        ),
        image_size=256,
        num_train_timesteps=num_train_timesteps,
        sampling_timesteps=sampling_timesteps,
        T1=t1,
        T2=t2,
    )

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

    k1 = int(round(t1 / num_train_timesteps * sampling_timesteps))
    k2 = int(round(t2 / num_train_timesteps * sampling_timesteps))
    stage_boundary = dict(
        t1=t1,
        t2=t2,
        k1=k1,
        k2=k2,
        sampling_timesteps=sampling_timesteps,
        num_train_timesteps=num_train_timesteps,
        stage_s3=sampling_timesteps - k2,
        stage_s2=k2 - k1,
        stage_s1=k1,
        data_root=str(Path(data_root)),
    )

    return dict(
        test_dataloader=test_dataloader,
        metric_list=metric_list,
        model=model,
        checkpoint_path_x3=checkpoint_path_x3,
        checkpoint_path_x2_x3=checkpoint_path_x2_x3,
        checkpoint_path_x1_x2_x3=checkpoint_path_x1_x2_x3,
        stage_boundary=stage_boundary,
    )


def export(globals_dict, **kwargs):
    globals_dict.update(build_config(**kwargs))
    globals_dict["__all__"] = [
        "test_dataloader",
        "metric_list",
        "model",
        "checkpoint_path_x3",
        "checkpoint_path_x2_x3",
        "checkpoint_path_x1_x2_x3",
        "stage_boundary",
    ]

