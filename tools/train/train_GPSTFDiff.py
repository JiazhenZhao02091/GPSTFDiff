# CUDA_VISIBLE_DEVICES="4" python tools/train/train_GPSTFDiff.py --congfig_path config/GPSTFDiff/syy_setting-9/CIA/config.py
# CUDA_VISIBLE_DEVICES="2" python tools/train/train_GPSTFDiff.py --congfig_path config/GPSTFDiff/syy_setting-9/CIA/config_adapter.py
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import importlib
import warnings

from src.utils import fix_random_seed
import sys
sys.path.insert(0, os.getcwd())
warnings.filterwarnings("ignore")

from src.trainer.GPSTFDiff_trianer_mask_loss import Trainer

parser = argparse.ArgumentParser(description="data generation")
parser.add_argument("--congfig_path", type=str, required=True)
parser.add_argument("--work_dir", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume_from", type=str, default=None)
parser.add_argument("--start_epoch", type=int, default=None)
args = parser.parse_args()

fix_random_seed(args.seed)
os.environ["GPSTFDIFF_SEED"] = str(args.seed)

congfig_path = args.congfig_path
config_module_string = congfig_path.replace("/", ".").replace(".py", "")
# exec(f'from {dataset_setting_config_module_string} import *')
config = importlib.import_module(config_module_string)
work_dir = args.work_dir or congfig_path.replace("config", "results").replace(".py", "")


exp = Trainer(
    congfig_path=congfig_path,
    train_root_dir_path=work_dir,
    train_dataloader=config.train_dataloader,
    val_dataloader=config.val_dataloader,
    model=config.model,
    optimizer=config.optimizer,
    scheduler=config.scheduler,
    metric_list=config.metric_list,
    train_params=getattr(config, "train_params", None),
)
if args.resume_from is not None:
    import torch

    checkpoint = torch.load(args.resume_from, map_location=exp.device)
    exp.model.load_state_dict(checkpoint["model"])
    exp.optimizer.load_state_dict(checkpoint["optimizer"])
    exp.ema.load_state_dict(checkpoint["ema"])
    exp.current_epoch = (
        args.start_epoch if args.start_epoch is not None else exp.current_epoch
    )
    exp.txt_logger.info(
        f"resume from {args.resume_from}, start_epoch: {exp.current_epoch}"
    )
exp.train()
