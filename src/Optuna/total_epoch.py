import torch
import optuna
import os
from tqdm import tqdm
from src.model.Laplasi.Four_stages_diffusion_inferencer import Diffusion
from src.model.LapSTFDiff import PredNoiseNetMKIRA_forward
from src.data.dataloader.data_sampler import EpochBasedSampler
from src.data.dataloader.worker_init import worker_init_fn
from functools import partial
from src.data.transforms import *
from src.data.dataset import SpatioTemporalFusionDataset
from torch.utils.data import DataLoader

# ==========================================
# Step 1: 初始化模型 (显存优化：共享实例)
# ==========================================
device = "cuda:1" if torch.cuda.is_available() else "cpu"

# 实例化一个共享的 UNet 实例以节省显存
shared_unet = PredNoiseNetMKIRA_forward(
    dim=128, channels=6, out_dim=6, 
    dim_mults=(1, 2, 4, 8), 
    use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0
)

diffusion_model = Diffusion(
    model_x4=shared_unet,
    model_x3_x4=shared_unet,
    model_x2_x3_x4=shared_unet,
    model_x1_x2_x3_x4=shared_unet,
    model=shared_unet,
    T1=100, T2=150, T3=300,
    image_size=256,
    sampling_timesteps=100,
)

checkpoint_path = "results_backup/LapSTFDiff/syy_setting-9/CIA/results_MKIRA_mult_4_dim_128/checkpoints/best_model_epoch_149.pth"

def get_model_ema_dict(path):
    if not os.path.exists(path): return None
    checkpoint = torch.load(path, map_location='cpu')
    raw_dict = checkpoint.get('model', checkpoint.get('ema', {}))
    return {k.replace('model.', '').replace('ema_model.', ''): v for k, v in raw_dict.items()}

# 加载一次权重，所有引用 shared_unet 的阶段都会同步
state_dict = get_model_ema_dict(checkpoint_path)
if state_dict:
    diffusion_model.model_x4.load_state_dict(state_dict, strict=False)

diffusion_model.to(device)

# ==========================================
# Step 2: 准备完整验证集数据
# ==========================================
train_transforms = [
    LoadData(key_list=['fine_img_01', 'fine_img_02', 'coarse_img_01', 'coarse_img_02']),
    RescaleToMinusOneOne(key_list=['fine_img_01', 'fine_img_02', 'coarse_img_01', 'coarse_img_02'], data_range=[0, 10000]),
    Format(key_list=['fine_img_01', 'fine_img_02', 'coarse_img_01', 'coarse_img_02']),
]

val_dataset = SpatioTemporalFusionDataset(
    dataset_name="CIA",
    data_root='data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/val',
    data_prefix_tmpl_dict=dict(fine_img_01='Landsat_01', fine_img_02='Landsat_02', coarse_img_01='MODIS_01', coarse_img_02='MODIS_02'),
    data_name_tmpl_dict=dict(fine_img_01='{}_L_{}', fine_img_02='{}_L_{}', coarse_img_01='{}_M_{}', coarse_img_02='{}_M_{}'),
    transform_func_list=train_transforms,
    is_serialize_data=True,
)

# 注意：为了评估准确，设置 shuffle=False
val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=1, 
    sampler=EpochBasedSampler(dataset=val_dataset, is_shuffle=False, seed=42),
    num_workers=4,
    worker_init_fn=partial(worker_init_fn, num_workers=4, rank=0, seed=42),
)

# ==========================================
# Step 3: 定义 Optuna 目标函数
# ==========================================
def objective(trial):
    # 1. 参数采样
    t1 = trial.suggest_int("T1", 50, 400)
    t2 = trial.suggest_int("T2", t1 + 20, 700)
    t3 = trial.suggest_int("T3", t2 + 20, 950)

    # 2. 参数注入模型并同步 Buffer
    diffusion_model.T1, diffusion_model.T2, diffusion_model.T3 = t1, t2, t3
    diffusion_model.init_para(timesteps=1000) 
    diffusion_model.to(device) # 必须将 init_para 新生成的 buffer 移到 GPU
    diffusion_model.eval()

    diffusion_model.sample_T1 = int(t1 / 1000 * diffusion_model.sampling_timesteps)
    diffusion_model.sample_T2 = int(t2 / 1000 * diffusion_model.sampling_timesteps)
    diffusion_model.sample_T3 = int(t3 / 1000 * diffusion_model.sampling_timesteps)

    # 3. 遍历整个验证集进行评估
    total_rmse = 0.0
    count = 0
    
    # 使用 tqdm 监控单个 Trial 的进度
    pbar = tqdm(val_dataloader, desc=f"Trial {trial.number}", leave=False)
    
    with torch.no_grad():
        for batch_data in pbar:
            # 获取数据
            c1 = batch_data['coarse_img_01'].to(device)
            c2 = batch_data['coarse_img_02'].to(device)
            f1 = batch_data['fine_img_01'].to(device)
            gt = batch_data['fine_img_02'].to(device)

            # 固定随机种子（对每一张图使用相同的初始噪声，保证 Trial 间公平对比）
            torch.manual_seed(42) 
            
            output = diffusion_model.sample(c1, c2, f1)
            
            # 还原到 [0, 1] 空间计算 RMSE
            output_norm = (output + 1.0) / 2.0
            gt_norm = (gt + 1.0) / 2.0
            
            mse = torch.mean((output_norm - gt_norm) ** 2)
            rmse = torch.sqrt(mse).item()
            
            total_rmse += rmse
            count += 1
            
            # 及时释放内存
            del output, c1, c2, f1, gt
            
            # 更新进度条显示当前平均 RMSE
            pbar.set_postfix({"curr_rmse": f"{total_rmse/count:.4f}"})

    avg_rmse = total_rmse / count
    
    # 汇报结果给 Optuna（支持剪枝判断）
    trial.report(avg_rmse, step=0) 
    
    torch.cuda.empty_cache()
    return avg_rmse

# ==========================================
# Step 4: 执行优化
# ==========================================
if __name__ == "__main__":
    db_path = "sqlite:///optuna_diffusion_study.db"
    
    study = optuna.create_study(
        study_name="diffusion_full_val_search",
        storage=db_path,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    print(f"\n[INFO] 正在对完整验证集（{len(val_dataset)}张图）进行调参。")
    print(f"[INFO] 监控命令: optuna-dashboard {db_path}")
    
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    print("\n" + "="*30)
    print("最优参数组合: ", study.best_params)
    print("最低均方根误差 (RMSE): ", study.best_value)
    print("="*30)