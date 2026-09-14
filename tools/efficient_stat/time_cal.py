import torch
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
from src.model.GPSTFDiff.diffusion_lap_inferencer import Diffusion  # GPSTFDiff
from src.model.GPSTFDiff import PredNoiseNetMKIRA
from src.model.GPSTFDiff.GPSTFDiff_ablation.net_no_FILM_inject import PredNoiseNetMKIRA as PredNoiseNetMKIRA_no_FiLM_inject  # GPSTFDiff no FiLM injection
from src.model.GPSTFDiff.GPSTFDiff_ablation.net_no_SA_CA import PredNoiseNetMKIRA as PredNoiseNetMKIRA_no_SA_CA  # GPSTFDiff no CA_SA
from src.model.FSDFormer import FSDFormer   # FSDFormer
from src.model.GPSTFDiff.diffusion_lap_inferencer_flops import Diffusion as Diffusion_flops
from src.model.GPSTFDiff.GPSTFDiff_ablation.net_no_dc import PredNoiseNetMKIRA as PredNoiseNetMKIRA_no_dc  # GPSTFDiff no DC

# 模拟计算单个 patch

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# ==========================================
# 【新增】日志配置
# ==========================================
log_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(log_dir, 'time_cal_abl_dc.log')

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
# 【新增】封装测速模块（内含 GPU Warm-up）
# ==========================================
def measure_inference_time(model_name, callable_func, runs=4, warmup=2):
    logging.info(f"正在测试 {model_name} ...")
    
    # 预热 (Warm-up) - 极其重要，否则第一次 CUDA 启动耗时会被算进去
    with torch.no_grad():
        for _ in range(warmup):
            callable_func()
            torch.cuda.synchronize()
    
    # 正式测速
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
# 初始化输入张量
# ==========================================
logging.info("初始化输入张量并传输至 GPU ...\n" + "-" * 30)
M1 = torch.randn(1, 6, 256, 256).cuda()
M2 = torch.randn(1, 6, 256, 256).cuda()
M3 = torch.randn(1, 6, 256, 256).cuda()
L1 = torch.randn(1, 6, 256, 256).cuda()
L2 = torch.randn(1, 6, 256, 256).cuda()
L3 = torch.randn(1, 6, 256, 256).cuda()

M_64 = torch.randn(1, 6, 64, 64).cuda()
M_128 = torch.randn(1, 6, 128, 128).cuda()

# ==========================================
# 开始测试各个模型
# ==========================================

# # 1. STFDCNN
# model = STFDCNN(6).cuda()
# measure_inference_time('STFDCNN', lambda: model(M1))
# del model; torch.cuda.empty_cache()

# # 2. STFGAN
# model = STFGANGenerator(6).cuda()
# measure_inference_time('STFGANGenerator', lambda: model(M_64, M_64, M_64, L1, L1))
# del model; torch.cuda.empty_cache()

# # 3. GANSTFM
# model = SFFusion().cuda()
# measure_inference_time('GANSTFMGenerator', lambda: model((M1, L1)))
# del model; torch.cuda.empty_cache()

# # 4. OPGAN
# model = OPGANGenerator().cuda()
# measure_inference_time('OPGANGenerator', lambda: model(M1, M2, L1))
# del model; torch.cuda.empty_cache()

# # 5. SwinSTFM
# model = SwinSTFM().cuda()
# measure_inference_time('SwinSTFM', lambda: model(M1, M2, L1))
# del model; torch.cuda.empty_cache()

# # 6. FSDFormer
# model = FSDFormer().cuda()
# measure_inference_time('FSDFormer', lambda: model(M1, M2, L1))
# del model; torch.cuda.empty_cache()

# # 7. STFMamba (如果在部分环境中未编译，用 try 保护防止中断)
# try:
#     model = model_STF().cuda()
#     measure_inference_time('STFMamba', lambda: model(M_128, M_128, M_128, 'cuda'))
#     del model; torch.cuda.empty_cache()
# except Exception as e:
#     logging.warning(f"STFMamba 测试跳过，因缺少环境依赖: {e}")
#     logging.info("-" * 30)

# # 8. STFDiff
# logging.info("注意：扩散模型测速使用 .sample() 包含多次迭代，耗时将显著多于 CNN 模型")
# model = GaussianDiffusion(
#     model=PredNoiseNet(dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4)),
#     image_size=256, timesteps=100, sampling_timesteps=10, objective="pred_x0", ddim_sampling_eta=0.0,
# ).cuda()
# measure_inference_time('STFDiff (sampling_timesteps=10)', lambda: model.sample(M1, M2, L2))
# del model; torch.cuda.empty_cache()

# 9. GPSTFDiff
# model = Diffusion(
#     model_x3 = PredNoiseNetMKIRA(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
#     model_x2_x3 = PredNoiseNetMKIRA(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
#     model_x1_x2_x3 = PredNoiseNetMKIRA(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
#     model = PredNoiseNetMKIRA(dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
#     T1=100, T2=150,
#     sampling_timesteps=10, image_size=256, # 这里暂时也按 10 步测速
# ).cuda()

# M1 = torch.randn(1, 6, 64, 64).cuda()
# measure_inference_time('GPSTFDiff (sampling_timesteps=10)', lambda: model.sample(M1, M1, M1) if hasattr(model, 'sample') else model(M1, M2, L1, L2)) 
# del model; torch.cuda.empty_cache()

model = Diffusion(
    model_x3 = PredNoiseNetMKIRA(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    model_x2_x3 = PredNoiseNetMKIRA(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    model_x1_x2_x3 = PredNoiseNetMKIRA(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    model = PredNoiseNetMKIRA(dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    T1=100, T2=200,
    sampling_timesteps=10, image_size=256, # 这里暂时也按 10 步测速
).cuda()

M1 = torch.randn(1, 6, 64, 64).cuda()
measure_inference_time('GPSTFDiff Full', lambda: model.sample(M1, M1, M1) if hasattr(model, 'sample') else model(M1, M2, L1, L2)) 
del model; torch.cuda.empty_cache()


model = Diffusion(
    model_x3 = PredNoiseNetMKIRA(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=False, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    model_x2_x3 = PredNoiseNetMKIRA(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=False, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    model_x1_x2_x3 = PredNoiseNetMKIRA(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=False, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    model = PredNoiseNetMKIRA(dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4), use_mkira=False, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    T1=100, T2=200,
    sampling_timesteps=10, image_size=256, # 这里暂时也按 10 步测速
).cuda()

# M1 = torch.randn(1, 6, 64, 64).cuda()
measure_inference_time('GPSTFDiff NoMKIRA', lambda: model.sample(M1, M1, M1) if hasattr(model, 'sample') else model(M1, M2, L1, L2)) 
del model; torch.cuda.empty_cache()

model = Diffusion(
    model_x3 = PredNoiseNetMKIRA_no_FiLM_inject(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=False, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    model_x2_x3 = PredNoiseNetMKIRA_no_FiLM_inject(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=False, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    model_x1_x2_x3 = PredNoiseNetMKIRA_no_FiLM_inject(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=False, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    model = PredNoiseNetMKIRA_no_FiLM_inject(dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4), use_mkira=False, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    T1=100, T2=200,
    sampling_timesteps=10, image_size=256, # 这里暂时也按 10 步测速
).cuda()

# M1 = torch.randn(1, 6, 64, 64).cuda()
measure_inference_time('GPSTFDiff NoFiLM', lambda: model.sample(M1, M1, M1) if hasattr(model, 'sample') else model(M1, M2, L1, L2)) 
del model; torch.cuda.empty_cache()

model = Diffusion(
    model_x3 = PredNoiseNetMKIRA_no_SA_CA(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=False, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    model_x2_x3 = PredNoiseNetMKIRA_no_SA_CA(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=False, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    model_x1_x2_x3 = PredNoiseNetMKIRA_no_SA_CA(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=False, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    model = PredNoiseNetMKIRA_no_SA_CA(dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4), use_mkira=False, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    T1=100, T2=200,
    sampling_timesteps=10, image_size=256, # 这里暂时也按 10 步测速
).cuda()

# M1 = torch.randn(1, 6, 64, 64).cuda()
measure_inference_time('GPSTFDiff NoSA-CA', lambda: model.sample(M1, M1, M1) if hasattr(model, 'sample') else model(M1, M2, L1, L2)) 
del model; torch.cuda.empty_cache()



model = Diffusion_flops(
    model_x3 = PredNoiseNetMKIRA_no_SA_CA(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0, include_f1_in_cond = False),
    model_x2_x3 = PredNoiseNetMKIRA_no_SA_CA(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0, include_f1_in_cond = False),
    model_x1_x2_x3 = PredNoiseNetMKIRA_no_SA_CA(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0, include_f1_in_cond = False),
    model = PredNoiseNetMKIRA_no_SA_CA(dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0, include_f1_in_cond = False),
    sampling_timesteps=100, image_size=256,
).cuda()

# M1 = torch.randn(1, 6, 64, 64).cuda()
measure_inference_time('GPSTFDiff NoF1', lambda: model.sample(M1, M1, M1) if hasattr(model, 'sample') else model(M1, M2, L1, L2)) 
del model; torch.cuda.empty_cache()



model = Diffusion_flops(
    model_x3 = PredNoiseNetMKIRA_no_dc(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0, include_f1_in_cond =True, include_dc_in_cond = False),
    model_x2_x3 = PredNoiseNetMKIRA_no_dc(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0, include_f1_in_cond = True, include_dc_in_cond = False),
    model_x1_x2_x3 = PredNoiseNetMKIRA_no_dc(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0, include_f1_in_cond = True, include_dc_in_cond = False),
    model = PredNoiseNetMKIRA_no_dc(dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0, include_f1_in_cond = True, include_dc_in_cond = False),
    sampling_timesteps=100, image_size=256,
).cuda()

# M1 = torch.randn(1, 6, 64, 64).cuda()
measure_inference_time('GPSTFDiff NoDC', lambda: model.sample(M1, M1, M1) if hasattr(model, 'sample') else model(M1, M2, L1, L2)) 
del model; torch.cuda.empty_cache()


model = Diffusion_flops(
    model_x3 = PredNoiseNetMKIRA_no_dc(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0, include_f1_in_cond =False, include_dc_in_cond = False),
    model_x2_x3 = PredNoiseNetMKIRA_no_dc(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0, include_f1_in_cond = False, include_dc_in_cond = False),
    model_x1_x2_x3 = PredNoiseNetMKIRA_no_dc(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0, include_f1_in_cond = False, include_dc_in_cond = False),
    model = PredNoiseNetMKIRA_no_dc(dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0, include_f1_in_cond = False, include_dc_in_cond = False),
    sampling_timesteps=100, image_size=256,
).cuda()

# M1 = torch.randn(1, 6, 64, 64).cuda()
measure_inference_time('GPSTFDiff No DC F1', lambda: model.sample(M1, M1, M1) if hasattr(model, 'sample') else model(M1, M2, L1, L2)) 
del model; torch.cuda.empty_cache()

logging.info(f"测试完毕。所有时间记录已保存至: {log_file}")

