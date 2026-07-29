import torch
import torch.nn.functional as F
import time
import logging
import os

from src.model import (
    SwinSTFM,           # SwinSTFM
    OPGANGenerator,     # OPGAN
    SFFusion,           # GANSTFM
    STFDCNN,            # STFDCNN
    STFGANGenerator,    # STFGAN
    GaussianDiffusion,  # STFDiff
    PredNoiseNet,
)
from src.model.stfmamba.model import model_STF # STFMamba
from src.model.LapSTFDiff.diffusion_lap_inferencer import Diffusion  # LapSTFDiff
from src.model.LapSTFDiff import PredNoiseNetMKIRA
from src.model.FSDFormer import FSDFormer   # FSDFormer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ==========================================
# 日志配置
# ==========================================
log_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(log_dir, 'time_cal_full_image_3.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def log_result(model_name, avg_time):
    logging.info(f"{model_name}")
    logging.info(f"Time = {avg_time:.4f} s")
    logging.info("-" * 30)

# ==========================================
# 封装测速模块
# ==========================================
def measure_inference_time(model_name, callable_func, runs=4, warmup=2):
    logging.info(f"正在测试 {model_name} (整图) ...")
    
    with torch.no_grad():
        for _ in range(warmup):
            callable_func()
            torch.cuda.synchronize()
    
    total_time = 0.0
    with torch.no_grad():
        for _ in range(runs):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            callable_func()
            
            torch.cuda.synchronize()
            total_time += (time.perf_counter() - start_time)
    
    avg_time = total_time / runs
    log_result(model_name, avg_time)

# ==========================================
# 针对 Transformer/Mamba 等模型的分块推理函数
# ==========================================
def patch_inference(model, inputs, patch_size=256, is_mamba=False):
    """
    通用分块推理函数，防止大图显存溢出。
    inputs: 包含需送入模型的张量 Tuple, 如 (M1, M2, L1)
    """
    b, c, h, w = inputs[0].shape
    out = torch.zeros(b, 6, h, w, device=inputs[0].device)  # 假定输出为6波段
    
    # 滑动窗口切块
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            i_end = min(i + patch_size, h)
            j_end = min(j + patch_size, w)
            
            patches = []
            for inp in inputs:
                patch = inp[:, :, i:i_end, j:j_end]
                # 边界补齐 (Padding)，使得输入的尺寸严格等于 patch_size
                pad_h = patch_size - patch.shape[2]
                pad_w = patch_size - patch.shape[3]
                if pad_h > 0 or pad_w > 0:
                    patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='reflect')
                patches.append(patch)
            
            # 前向传播
            if is_mamba:
                pred = model(*patches, 'cuda')
            else:
                pred = model(*patches)
                
            if isinstance(pred, tuple):
                pred = pred[0]
                
            # 去掉 Padding 的部分，拼接回原图
            out[:, :, i:i_end, j:j_end] = pred[:, :, :i_end-i, :j_end-j]
            
    return out

# ==========================================
# 初始化整图输入张量
# ==========================================
logging.info("初始化整图输入张量并传输至 GPU ...\n" + "-" * 30)

B, C, H, W = 1, 6, 1792, 1280

# 1. 原始大小 (Landsat 以及大部分同分辨率的输入)
M1_full = torch.randn(B, C, H, W).cuda()
M2_full = torch.randn(B, C, H, W).cuda()
M3_full = torch.randn(B, C, H, W).cuda()
L1_full = torch.randn(B, C, H, W).cuda()
L2_full = torch.randn(B, C, H, W).cuda()
L3_full = torch.randn(B, C, H, W).cuda()

# 2. STFGAN 所需的缩小 4 倍的 MODIS 输入
H_64, W_64 = H // 4, W // 4  # 448, 320
M_stfgan_full = torch.randn(B, C, H_64, W_64).cuda()

# ==========================================
# 开始整图推理测试
# ==========================================

# # 1. STFDCNN (直接送入整图)
# model = STFDCNN(6).cuda()
# measure_inference_time('STFDCNN', lambda: model(M1_full))
# del model; torch.cuda.empty_cache()

# # 2. STFGAN (直接送入相应分辨率的整图)
# model = STFGANGenerator(6).cuda()
# measure_inference_time('STFGANGenerator', lambda: model(M_stfgan_full, M_stfgan_full, M_stfgan_full, L1_full, L1_full))
# del model; torch.cuda.empty_cache()

# # 3. GANSTFM
# model = SFFusion().cuda()
# measure_inference_time('GANSTFMGenerator', lambda: model((M1_full, L1_full)))
# del model; torch.cuda.empty_cache()

# # 4. OPGAN
# model = OPGANGenerator().cuda()
# measure_inference_time('OPGANGenerator', lambda: model(M1_full, M2_full, L1_full))
# del model; torch.cuda.empty_cache()

# # 5. SwinSTFM (利用分块推理防止显存溢出)
# model = SwinSTFM().cuda()
# measure_inference_time('SwinSTFM', lambda: patch_inference(model, (M1_full, M2_full, L1_full), patch_size=256))
# del model; torch.cuda.empty_cache()

# # 6. FSDFormer (利用分块推理)
# model = FSDFormer().cuda()
# measure_inference_time('FSDFormer', lambda: patch_inference(model, (M1_full, M2_full, L1_full), patch_size=256))
# del model; torch.cuda.empty_cache()

# 7. STFMamba
try:
    model = model_STF().cuda()
    # 针对 Mamba 通常按训练阶段使用较小 patch 进行滑动预测（例如 128x128）
    measure_inference_time('STFMamba', lambda: patch_inference(model, (M1_full, M2_full, L1_full), patch_size=128, is_mamba=True))
    del model; torch.cuda.empty_cache()
except Exception as e:
    logging.warning(f"STFMamba 测试跳过，因缺少环境依赖: {e}")
    logging.info("-" * 30)

# # 8. STFDiff
# logging.info("注意：扩散模型整图测速耗时较长")
# model = GaussianDiffusion(
#     model=PredNoiseNet(dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4)),
#     image_size=256, timesteps=100, sampling_timesteps=10, objective="pred_x0", ddim_sampling_eta=0.0,
# ).cuda()
# measure_inference_time('STFDiff (sampling_timesteps=10)', lambda: model.sample(M1_full, M2_full, L2_full))
# del model; torch.cuda.empty_cache()

# # 9. LapSTFDiff
# model = Diffusion(
#     model_x3 = PredNoiseNetMKIRA(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
#     model_x2_x3 = PredNoiseNetMKIRA(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
#     model_x1_x2_x3 = PredNoiseNetMKIRA(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
#     model = PredNoiseNetMKIRA(dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
#     T1=100, T2=150,
#     sampling_timesteps=10, image_size=256, 
# ).cuda()

# measure_inference_time('LapSTFDiff (sampling_timesteps=10)', lambda: model.sample(M1_full, M1_full, M1_full) if hasattr(model, 'sample') else model(M1_full, M2_full, L1_full, L2_full)) 
# del model; torch.cuda.empty_cache()

logging.info(f"测试完毕。所有时间记录已保存至: {log_file}")