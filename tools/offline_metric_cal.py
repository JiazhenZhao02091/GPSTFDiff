from pathlib import Path
from src.metrics import *
from src.logger import FusionLogger
import tifffile as tiff
import numpy as np
import torch

gt_dir_path = Path(
    'data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/test/patch/Landsat_02'
)

# pred_dir_path_stf = "results/stfdcnn/syy_setting~9/ML/inference~stage_1_SGD_1e-2~stage_2_SGD_1e-1/full/imgs/stage_2/ML/save_img"
# pred_dir_path_stf = "results/stfgan/syy_setting-9/ML/inference~stage_1~RMSProp~stage_2~RMSProp/full/imgs/stage_2/ML/save_img"


# pred_dir_path_stf = "results/swinstf/syy_setting-9/ML/inference~Adam_1e-4~StepLR_step_size-15_gamma_5e-1/full/imgs/ML/save_img"
# pred_dir_path_stf = "results/LapSTFDiff/lap/syy_setting-9/ML/inference/full/imgs/ML/0/save_img"
# pred_dir_path_stf = "results/stfmamba/syy_setting-9/ML/inferencer/full/imgs/ML/save_img"
# pred_dir_path_stf = "results/stfdiff/syy_setting-9/model6_GN_SiLU/ML/inference/full/imgs/ML/0/save_img"
# pred_dir_path_stf = "results/opgan/syy_setting-9/ML/inference~RMSProp/full/imgs/ML/save_img"
# pred_dir_path_stf = "results/ganstfm/syy_setting-9/ML/inference~Adam_1e-4/full/imgs/ML/save_img"
# pred_dir_path_stf = "results/fsdformer/syy_setting-9/ML/inferencer/full/imgs/ML/save_img"


# pred_dir_path_stf = "results/starfm/syy_setting~9/ML/one_pair~patch_size_120~patch_stride_50~window_size_51~num_classes_20/patch/imgs/ML/save_img"

# pred_dir_path_stf = "results/FSDAF/ML/patch"

pred_dir_path_stf = "results/LapSTFDiff/lap/syy_setting-9/CIA/inference/inferency_8/patch/imgs/CIA/0/save_img"

pred_dir_path = Path(pred_dir_path_stf)
exp_name = "LapSTFDiff_patch"

# metric_list = [
#     RMSE(),
#     MAE(),
#     PSNR(max_value=1.0),
#     SSIM(data_range=1.0),
#     ERGAS(ratio=1.0 / 16.0),
#     CC(),
#     SAM(),
#     UIQI(),
# ]

metric_list = [
    RMSE(is_reduce_channel=False),
    MAE(is_reduce_channel=False),
    # PSNR(max_value=1.0, is_reduce_channel=False),
    PSNRONE(max_value=1.0, is_reduce_channel=False),
    SSIM(data_range=1.0, is_reduce_channel=False),
    ERGAS(ratio=1.0 / 16.0),
    CC(is_reduce_channel=False),
    SAM(),
    UIQI( is_reduce_channel=False),
]



txt_logger = FusionLogger(
    logger_name='offline_metric_cal',
    log_level='INFO',
    log_file=f'offline_metric_cal_{exp_name}.log'
)

gt_img_path_list = sorted(list(gt_dir_path.glob('*.tif')))
pred_img_path_list = sorted(list(pred_dir_path.glob('*.tif')))

assert len(gt_img_path_list) == len(pred_img_path_list)
img_num = len(gt_img_path_list)

# 1. 初始化一个字典来存储所有图片的结果
all_results = {metric.__name__: [] for metric in metric_list}

for i in range(img_num):
    gt_img_path = gt_img_path_list[i]

    pred_img_path = pred_img_path_list[i]

    gt_img = tiff.imread(str(gt_img_path))
    pred_img = tiff.imread(str(pred_img_path))
    # 真对FitFC
    # pred_img = pred_img.transpose(1, 2, 0)  # (Bands, H, W) -> (H, W, Bands)
    
    #TODO: ML 不需要进行除
    gt_img = gt_img.astype(np.float32) / 10000.0
    pred_img = pred_img.astype(np.float32) / 10000.0
    # gt_img = gt_img.astype(np.float32)
    # pred_img = pred_img.astype(np.float32)

    gt_img = torch.from_numpy(gt_img.transpose(2, 0, 1)).unsqueeze(0)
    pred_img = torch.from_numpy(pred_img.transpose(2, 0, 1)).unsqueeze(0)

    msg = f'evaluate {i + 1}/{img_num}'
    for metric in metric_list:
        metric_name = metric.__name__
        metric_value = metric(gt_img, pred_img)

        # 2. 将当前图片的结果存入列表 (转为 numpy 方便后续计算)
        if isinstance(metric_value, torch.Tensor):
            val_numpy = metric_value.cpu().numpy()
        else:
            val_numpy = np.array(metric_value)
        all_results[metric_name].append(val_numpy)
        

        # 修改 2: 判断返回值是标量还是向量，分别处理
        if isinstance(metric_value, torch.Tensor) and metric_value.numel() > 1:
            # 如果是多波段结果 (Vector)
            avg_val = metric_value.mean().item() # 手动计算平均值
            band_vals = metric_value.tolist()
            # 格式化为: 指标名: 平均值 (波段1/波段2/...)
            band_str = "\t".join([f"{v:.4f}" for v in band_vals])
            msg += f', {metric_name}: {avg_val:.4f} ({band_str})'
        else:
            # 如果是全局结果 (Scalar)
            val = metric_value.item() if isinstance(metric_value, torch.Tensor) else metric_value
            msg += f', {metric_name}: {val:.4f}'

        # msg += f', {metric_name}: {metric_value:.4f}'
    txt_logger.info(msg)

# 3. 循环结束后，计算并输出所有图片的平均值
txt_logger.info("-" * 100)
avg_msg = f'Average over {img_num} images'

for metric in metric_list:
    metric_name = metric.__name__
    # 将列表堆叠成数组 (N, ) 或 (N, Bands)
    values = np.array(all_results[metric_name])
    # 对第一维度 (图片数量) 求平均
    mean_val = np.mean(values, axis=0)
    
    if mean_val.size > 1:
        # 如果是多波段，计算总平均和各波段平均
        avg_global = np.mean(mean_val)
        band_str = "\t".join([f"{v:.4f}" for v in mean_val])
        avg_msg += f', {metric_name}: {avg_global:.4f} ({band_str})'
    else:
        # 如果是单值
        avg_msg += f', {metric_name}: {float(mean_val):.4f}'
txt_logger.info(avg_msg)
