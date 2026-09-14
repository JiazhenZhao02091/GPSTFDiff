import torch
import logging
import os
from thop import profile
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
from src.model.GPSTFDiff.resnet_MKIRA_simple_forward import PredNoiseNetMKIRA_forward
from src.model.GPSTFDiff.GPSTFDiff_ablation.net_no_FILM_inject import PredNoiseNetMKIRA as PredNoiseNetMKIRA_no_FiLM_inject  # GPSTFDiff no FiLM injection
from src.model.GPSTFDiff.GPSTFDiff_ablation.net_no_SA_CA import PredNoiseNetMKIRA as PredNoiseNetMKIRA_no_SA_CA  # GPSTFDiff no CA_SA
from src.model.FSDFormer import FSDFormer   # FSDFormer
from src.model.GPSTFDiff.diffusion_lap_inferencer_flops import Diffusion as Diffusion_flops
from src.model.GPSTFDiff.GPSTFDiff_ablation.net_no_dc import PredNoiseNetMKIRA as PredNoiseNetMKIRA_no_dc  # GPSTFDiff no DC
# 这是 FLOPs 和 Params 统计，不是时间

# ==========================================
# 【新增】日志配置
# ==========================================
log_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(log_dir, 'param_flops_abl_dc.log')

# 设置 logging 既输出到控制台，又写入文件
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'), # 每次运行覆写日志，若追加可用 'a'
        logging.StreamHandler()
    ]
)

# 封装一个辅助打印函数，让代码更整洁
def log_result(model_name, flops, params):
    logging.info(f"{model_name}")
    logging.info(f"FLOPs = {flops / 1000**3:.4f} G")
    logging.info(f"Params = {params / 1000**2:.4f} M")
    logging.info("-" * 30)

# ==========================================

M1 = torch.randn(1, 6, 256, 256)
M2 = torch.randn(1, 6, 256, 256)
M3 = torch.randn(1, 6, 256, 256)
L1 = torch.randn(1, 6, 256, 256)
L2 = torch.randn(1, 6, 256, 256)
L3 = torch.randn(1, 6, 256, 256)


# logging.info("开始测试 STFDCNN ...")
# model = STFDCNN(6)
# flops, params = profile(model, inputs=(M1,), verbose=False) # verbose=False可以关闭 thop 的底层日志轰炸
# log_result('STFDCNN', flops, params)


# logging.info("开始测试 STFGAN ...")
# M = torch.randn(1, 6, 64, 64)
# model = STFGANGenerator(6)
# flops, params = profile(model, inputs=(M, M, M, L3, L1,), verbose=False)
# log_result('STFGANGenerator', flops, params)


# logging.info("开始测试 GANSTFM ...")
# model = SFFusion()
# flops, params = profile(model, inputs=((M1, L1),), verbose=False)
# log_result('GANSTFMGenerator', flops, params)


# logging.info("开始测试 OPGAN ...")
# model = OPGANGenerator()
# flops, params = profile(model, inputs=(M1, M2, L1,), verbose=False)
# log_result('OPGANGenerator', flops, params)


# logging.info("开始测试 SwinSTFM ...")
# model = SwinSTFM()
# flops, params = profile(model, inputs=(M1, M2, L1,), verbose=False)
# log_result('SwinSTFM', flops, params)


# logging.info("开始测试 FSDFormer ...")
# model = FSDFormer()
# flops, params = profile(model, inputs=(M1, M2, L1,), verbose=False)
# log_result('FSDFormer', flops, params)


# logging.info("开始测试 STFMamba ...")
# model = model_STF().cuda()
# M_cuda = torch.randn(1, 6, 128, 128).cuda()
# flops, params = profile(model, inputs=(M_cuda, M_cuda, M_cuda, 'cuda',), verbose=False)
# log_result('STFMamba', flops, params)


# logging.info("开始测试 STFDiff ...")
# model = GaussianDiffusion(
#     model=PredNoiseNet(dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4)),
#     image_size=256, timesteps=100, sampling_timesteps=50, objective="pred_x0", ddim_sampling_eta=0.0,
# )
# flops, params = profile(model, inputs=(M1, M2, L1, L2,), verbose=False)
# log_result('STFDiff', flops, params)


logging.info("开始测试 GPSTFDiff Full ...")
model = Diffusion(
    model_x3 = PredNoiseNetMKIRA_forward(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    model_x2_x3 = PredNoiseNetMKIRA_forward(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    model_x1_x2_x3 = PredNoiseNetMKIRA_forward(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    model = PredNoiseNetMKIRA_forward(dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    sampling_timesteps=100, image_size=256,
)
flops, params = profile(model, inputs=(M1, M2, L1, L2,), verbose=False)
log_result('GPSTFDiff_full', flops, params)

logging.info("开始测试 GPSTFDiff No AdaMDR...")
T = torch.randint(0, 1000, (1,))
model = Diffusion(
    model_x3 = PredNoiseNetMKIRA_forward(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=False, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    model_x2_x3 = PredNoiseNetMKIRA_forward(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=False, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    model_x1_x2_x3 = PredNoiseNetMKIRA_forward(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=False, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    model = PredNoiseNetMKIRA_forward(dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4), use_mkira=False, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    sampling_timesteps=100, image_size=256,
)
flops, params = profile(model, inputs=(M1, M2, L1, L2,), verbose=False)
log_result('GPSTFDiff_no_adaMDR', flops, params)

logging.info("开始测试 GPSTFDiff No FiLM injection...")
model = Diffusion_flops(
    model_x3 = PredNoiseNetMKIRA_no_FiLM_inject(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    model_x2_x3 = PredNoiseNetMKIRA_no_FiLM_inject(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    model_x1_x2_x3 = PredNoiseNetMKIRA_no_FiLM_inject(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    model = PredNoiseNetMKIRA_no_FiLM_inject(dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    sampling_timesteps=100, image_size=256,
)
flops, params = profile(model, inputs=(M1, M2, L1, L2, ), verbose=False)
log_result('GPSTFDiff_no_FiLM_injection', flops, params)


logging.info("开始测试 GPSTFDiff No CA_SA...")
model = Diffusion_flops(
    model_x3 = PredNoiseNetMKIRA_no_SA_CA(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    model_x2_x3 = PredNoiseNetMKIRA_no_SA_CA(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    model_x1_x2_x3 = PredNoiseNetMKIRA_no_SA_CA(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    model = PredNoiseNetMKIRA_no_SA_CA(dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0),
    sampling_timesteps=100, image_size=256,
)
flops, params = profile(model, inputs=(M1, M2, L1, L2, ), verbose=False)
log_result('GPSTFDiff_no_CA_SA', flops, params)

logging.info("开始测试 GPSTFDiff No f1...")
model = Diffusion_flops(
    model_x3 = PredNoiseNetMKIRA_no_SA_CA(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0, include_f1_in_cond = False),
    model_x2_x3 = PredNoiseNetMKIRA_no_SA_CA(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0, include_f1_in_cond = False),
    model_x1_x2_x3 = PredNoiseNetMKIRA_no_SA_CA(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0, include_f1_in_cond = False),
    model = PredNoiseNetMKIRA_no_SA_CA(dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0, include_f1_in_cond = False),
    sampling_timesteps=100, image_size=256,
)
flops, params = profile(model, inputs=(M1, M2, L1, L2, ), verbose=False)
log_result('GPSTFDiff_no_f1', flops, params)


logging.info("开始测试 GPSTFDiff No dc...")
model = Diffusion_flops(
    model_x3 = PredNoiseNetMKIRA_no_dc(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0, include_f1_in_cond =True, include_dc_in_cond = False),
    model_x2_x3 = PredNoiseNetMKIRA_no_dc(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0, include_f1_in_cond = True, include_dc_in_cond = False),
    model_x1_x2_x3 = PredNoiseNetMKIRA_no_dc(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0, include_f1_in_cond = True, include_dc_in_cond = False),
    model = PredNoiseNetMKIRA_no_dc(dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0, include_f1_in_cond = True, include_dc_in_cond = False),
    sampling_timesteps=100, image_size=256,
)
flops, params = profile(model, inputs=(M1, M2, L1, L2, ), verbose=False)
log_result('GPSTFDiff_no_dc', flops, params)

logging.info("开始测试 GPSTFDiff No dc f1...")
model = Diffusion_flops(
    model_x3 = PredNoiseNetMKIRA_no_dc(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0, include_f1_in_cond =False, include_dc_in_cond = False),
    model_x2_x3 = PredNoiseNetMKIRA_no_dc(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0, include_f1_in_cond = False, include_dc_in_cond = False),
    model_x1_x2_x3 = PredNoiseNetMKIRA_no_dc(dim=128, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0, include_f1_in_cond = False, include_dc_in_cond = False),
    model = PredNoiseNetMKIRA_no_dc(dim=64, channels=6, out_dim=6, dim_mults=(1, 2, 4), use_mkira=True, mkira_up_indices=(1, 2), mkira_init_alpha=0.0, include_f1_in_cond = False, include_dc_in_cond = False),
    sampling_timesteps=100, image_size=256,
)
flops, params = profile(model, inputs=(M1, M2, L1, L2, ), verbose=False)
log_result('GPSTFDiff_no_dc_f1', flops, params)


logging.info(f"测试完毕。所有结果已保存至: {log_file}")

