import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ['OMP_NUM_THREADS'] = '1'

# from src.inferencer.dit_inferencer import Inferencer
from src.inferencer.GPSTFDiff_lap_inferencer import Inferencer
import argparse
import importlib
import warnings
from src.utils import fix_random_seed
import sys
sys.path.insert(0, os.getcwd())
warnings.filterwarnings("ignore")

fix_random_seed(42)
parser = argparse.ArgumentParser(description="data generation")
parser.add_argument("--congfig_path", type=str, required=True)
args = parser.parse_args()

congfig_path = args.congfig_path
config_module_string = congfig_path.replace("/", ".").replace(".py", "")
config = importlib.import_module(config_module_string)
work_dir = congfig_path.replace("config", "results").replace(".py", "/")
img_mode = config.test_dataloader.dataset.data_root.name
work_dir = work_dir + img_mode

exp = Inferencer(
    congfig_path=congfig_path,
    inference_root_dir_path=work_dir,
    test_dataloader=config.test_dataloader,
    model=config.model,
    checkpoint_path_x1_x2_x3=config.checkpoint_path_x1_x2_x3,
    checkpoint_path_x2_x3=config.checkpoint_path_x2_x3,
    checkpoint_path_x3=config.checkpoint_path_x3,
    metric_list=config.metric_list,
)

exp.inference()