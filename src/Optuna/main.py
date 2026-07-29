import torch
import optuna
from tqdm import tqdm
from src.model.Laplasi.Four_stages_diffusion_inferencer import Diffusion
from src.model.LapSTFDiff import PredNoiseNetMKIRA_forward
from src.data.dataloader.data_sampler import EpochBasedSampler
from src.data.dataloader.worker_init import worker_init_fn
from functools import partial
from src.data.transforms import *
from src.data.dataset import SpatioTemporalFusionDataset
from torch.utils.data import DataLoader, ConcatDataset


# --- Step 1: 初始化你的所有模型 ---
device = "cuda:1" if torch.cuda.is_available() else "cpu"

diffusion_model = Diffusion(
    model_x4= PredNoiseNetMKIRA_forward(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    model_x3_x4 = PredNoiseNetMKIRA_forward(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    model_x2_x3_x4 = PredNoiseNetMKIRA_forward(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    model_x1_x2_x3_x4 = PredNoiseNetMKIRA_forward(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    model = PredNoiseNetMKIRA_forward(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    T1= 100,
    T2= 150,
    T3= 300,
    image_size=256,
    sampling_timesteps=100,
)

checkpoint_path_x4          = "results_backup/LapSTFDiff/syy_setting-9/CIA/results_MKIRA_mult_4_dim_128/checkpoints/best_model_epoch_149.pth"
checkpoint_path_x3_x4       = "results_backup/LapSTFDiff/syy_setting-9/CIA/results_MKIRA_mult_4_dim_128/checkpoints/best_model_epoch_149.pth"
checkpoint_path_x2_x3_x4    = "results_backup/LapSTFDiff/syy_setting-9/CIA/results_MKIRA_mult_4_dim_128/checkpoints/best_model_epoch_149.pth"
checkpoint_path_x1_x2_x3_x4 = "results_backup/LapSTFDiff/syy_setting-9/CIA/results_MKIRA_mult_4_dim_128/checkpoints/best_model_epoch_149.pth"

def get_model_ema_dict(checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    raw_model_dict = checkpoint.get('model', None)
    model_dict = {}
    if raw_model_dict:
        for k, v in raw_model_dict.items():
            new_k = k.replace('model.', '') if k.startswith('model.') else k
            model_dict[new_k] = v

    raw_ema_dict = checkpoint.get('ema', {})
    ema_dict = {}
    for key, value in raw_ema_dict.items():
        if key.startswith('ema_model.'):
            clean_key = key.replace('ema_model.', '', 1)
            ema_dict[clean_key] = value
        elif key.startswith('model.'):
            clean_key = key.replace('model.', '', 1)
            ema_dict[clean_key] = value
        else:
            ema_dict[key] = value
            
    return model_dict, ema_dict

model_dict_x4, ema_dict_x4 = get_model_ema_dict(checkpoint_path_x4) if checkpoint_path_x4 else (None, None)
model_dict_x3_x4, ema_dict_x3_x4 = get_model_ema_dict(checkpoint_path_x3_x4) if checkpoint_path_x3_x4 else (None, None)
model_dict_x2_x3_x4, ema_dict_x2_x3_x4 = get_model_ema_dict(checkpoint_path_x2_x3_x4) if checkpoint_path_x2_x3_x4 else (None, None)
model_dict_x1_x2_x3_x4, ema_dict_x1_x2_x3_x4 = get_model_ema_dict(checkpoint_path_x1_x2_x3_x4) if checkpoint_path_x1_x2_x3_x4 else (None, None)

missing, unexpected = diffusion_model.model_x4.load_state_dict(model_dict_x4, strict=False)
missing, unexpected = diffusion_model.model_x3_x4.load_state_dict(model_dict_x3_x4, strict=False)
missing, unexpected = diffusion_model.model_x2_x3_x4.load_state_dict(model_dict_x2_x3_x4, strict=False)
missing, unexpected = diffusion_model.model_x1_x2_x3_x4.load_state_dict(model_dict_x1_x2_x3_x4, strict=False)


diffusion_model.to(device)

# --- Step 2: 准备固定的测试数据 (1-2组即可) ---
# 建议直接从你的 Dataloader 里拿出一组存成 Tensor
# inputs = {"c1": coarse_01, "c2": coarse_02, "f1": fine_01, "gt": gt}

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

train_transforms_key_list = [
    'fine_img_01',
    'fine_img_02',
    'coarse_img_01',
    'coarse_img_02',
]

train_transforms = [
    LoadData(key_list=train_transforms_key_list),
    RescaleToMinusOneOne(key_list=train_transforms_key_list, data_range=[0, 10000]),
    Format(key_list=train_transforms_key_list),
]

val_dataset = dataset_cls_func(
            dataset_name = "CIA",
            data_root='data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/val',
            transform_func_list=train_transforms,
        )
val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=1,
    sampler=EpochBasedSampler(dataset=val_dataset, is_shuffle=False, seed=42),
    num_workers=4,
    worker_init_fn=partial(worker_init_fn, num_workers=4, rank=0, seed=42),
)

# 提取两组固定样本
inputs_list = []
for batch_idx, batch_data in enumerate(val_dataloader):
    if batch_idx in [0, 2]:
        inputs_list.append({
            'c1': batch_data['coarse_img_01'].to(device),
            'c2': batch_data['coarse_img_02'].to(device),
            'f1': batch_data['fine_img_01'].to(device),
            'gt': batch_data['fine_img_02'].to(device)
        })
    if len(inputs_list) == 2: break

def objective(trial):
    # 1. 参数采样：确保 T1 < T2 < T3
    t1 = trial.suggest_int("T1", 50, 400)
    t2 = trial.suggest_int("T2", t1 + 20, 700)
    t3 = trial.suggest_int("T3", t2 + 20, 950)

    # 2. 参数注入模型
    diffusion_model.T1, diffusion_model.T2, diffusion_model.T3 = t1, t2, t3
    
    # [关键修复 1] 重新生成参数表（此时新生成的 buffer 在 CPU 上）
    diffusion_model.init_para(timesteps=1000) 
    
    # [关键修复 2] 将更新后的 buffer 同步回 GPU，否则会报错
    diffusion_model.to(device)
    
    # [关键修复 3] 确保模型处于验证模式（关闭 Dropout，固定 Norm）
    diffusion_model.eval()

    # 更新采样节点
    diffusion_model.sample_T1 = int(t1 / 1000 * diffusion_model.sampling_timesteps)
    diffusion_model.sample_T2 = int(t2 / 1000 * diffusion_model.sampling_timesteps)
    diffusion_model.sample_T3 = int(t3 / 1000 * diffusion_model.sampling_timesteps)

    # 3. 推理（固定噪声种子非常重要！）
    torch.manual_seed(42) 
    total_rmse = 0
    with torch.no_grad():
        for inp in inputs_list:
            torch.manual_seed(42) # 关键：每次 sample 前固定种子
            output = diffusion_model.sample(inp['c1'], inp['c2'], inp['f1'])
            
            # 还原到 [0, 1] 空间计算 RMSE
            output_norm = (output + 1.0) / 2.0
            gt_norm = (inp['gt'] + 1.0) / 2.0
            
            mse = torch.mean((output_norm - gt_norm) ** 2)
            total_rmse += torch.sqrt(mse).item()
            
            del output # 及时释放
            
    avg_rmse = total_rmse / len(inputs_list)
    # 显存清理
    torch.cuda.empty_cache()
    return avg_rmse

# # --- Step 3: 执行优化 ---
if __name__ == "__main__":
    # 使用 SQLite 存储，方便使用 optuna-dashboard 监控
    db_path = "sqlite:///optuna_diffusion_study.db"
    
    study = optuna.create_study(
        study_name="diffusion_T_search_v1",
        storage=db_path,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    print(f"\n[INFO] 优化开始。请在另一个终端运行: optuna-dashboard {db_path} 来实时监控进度。")
    
    # 开始暴力搜索
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    print("\n" + "="*30)
    print("最优参数组合: ", study.best_params)
    print("最低均方根误差 (RMSE): ", study.best_value)
    print("="*30)