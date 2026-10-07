import importlib
import os
import warnings

from src.trainer.spstfm_trainer import Trainer
from src.utils import fix_random_seed
import argparse

warnings.filterwarnings("ignore")

os.environ['OMP_NUM_THREADS'] = '4'
import sys
sys.path.insert(0, os.getcwd())
fix_random_seed(42)

parser = argparse.ArgumentParser(description='data generation')
parser.add_argument('--congfig_path', type=str, required=True)
args = parser.parse_args()

congfig_path = args.congfig_path
config_module_string = congfig_path.replace('/', '.').replace('.py', '')

config = importlib.import_module(config_module_string)
work_dir = congfig_path.replace('config', 'results').replace('.py', '')


exp = Trainer(
    congfig_path=congfig_path,
    train_root_dir_path=work_dir,
    train_dataloader=config.train_dataloader,
    model=config.model,
)
exp.train()
