# ========================       Hot       ======================================
# python tools/train/train_dinostf.py --congfig_path config/dinostf/syy_setting-9/CIA/config.py
# python tools/train/train_dinostf.py --congfig_path config/dinostf/syy_setting-9/LGC/config.py
# ========================       Frozen    =======================================
# python tools/train/train_dinostf.py --congfig_path config/dinostf/syy_setting-9/froze/CIA/config.py
# ========================  PatchEmbedding =======================================
# python tools/train/train_dinostf.py --congfig_path config/dinostf/syy_setting-9/dino_patchembedding/hot/CIA/config.py
# ========================  Position =======================================
# python tools/train/train_dinostf.py --congfig_path config/dinostf/syy_setting-9/dino_position/hot/config.py
# ========================  Activation =======================================
# python tools/train/train_dinostf.py --congfig_path config/dinostf/syy_setting-9/dino_activation/hot/config.py
# ========================  PatchEmbedding + Activation + Position =======================================
# python tools/train/train_dinostf.py --congfig_path config/dinostf/syy_setting-9/dino_patch_position_activation/hot/CIA/config.py


# python tools/train/train_dinostf.py --congfig_path config/dinostf/syy_setting-9/dino_patchembedding/hot/CIA/config.py
# python tools/train/train_dinostf.py --congfig_path config/dinostf/syy_setting-9/dino_position/hot/config.py

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'   # 每个进程只使用一个线程进行计算

import argparse
import importlib
import warnings

from src.trainer.dinostf_trainer import Trainer
from src.utils import fix_random_seed
import sys
sys.path.insert(0, os.getcwd())

warnings.filterwarnings("ignore")

fix_random_seed(42)

parser = argparse.ArgumentParser(description='data generation')
parser.add_argument('--congfig_path', type=str, required=True)
args = parser.parse_args()

congfig_path = args.congfig_path
config_module_string = congfig_path.replace('/', '.').replace('.py', '')
# exec(f'from {dataset_setting_config_module_string} import *')
config = importlib.import_module(config_module_string)
work_dir = congfig_path.replace('config', 'results').replace('.py', '')


exp = Trainer(
    congfig_path=congfig_path,
    train_root_dir_path=work_dir,
    train_dataloader=config.train_dataloader,
    val_dataloader=config.val_dataloader,
    model=config.model,
    optimizer=config.optimizer,
    scheduler=config.scheduler,
    metric_list=config.metric_list,
)
exp.train()
