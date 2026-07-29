"""
    - python tools/inference/test_dinostf.py --congfig_path config/dinostf/syy_setting-9/CIA/inference.py
    - python tools/inference/test_dinostf.py --congfig_path config/dinostf/syy_setting-9/LGC/inference.py
    HOT
    - python tools/inference/test_dinostf.py --congfig_path config/dinostf/syy_setting-9/hot/CIA/inference.py
    Fronzen
    - python tools/inference/test_dinostf.py --congfig_path config/dinostf/syy_setting-9/froze/CIA/inference.py
    - python tools/inference/test_dinostf.py --congfig_path config/dinostf/syy_setting-9/froze/inital/inference.py
    PatchEmbedding
    python tools/inference/test_dinostf.py --congfig_path config/dinostf/syy_setting-9/dino_patchembedding/hot/CIA/inference.py
    Activation
    python tools/inference/test_dinostf.py --congfig_path config/dinostf/syy_setting-9/dino_activation/hot/inference.py
    Position
    python tools/inference/test_dinostf.py --congfig_path config/dinostf/syy_setting-9/dino_position/hot/inference.py
    All in 
    python tools/inference/test_dinostf.py --congfig_path config/dinostf/syy_setting-9/dino_patch_position_activation/hot/CIA/inference.py


    python tools/inference/test_dinostf.py  --congfig_path config/dinostf/syy_setting-9/hot/tmp/inference.py
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['OMP_NUM_THREADS'] = '1'

from src.inferencer.dinostf_inferencer import Inferencer
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
# exec(f'from {dataset_setting_config_module_string} import *')
config = importlib.import_module(config_module_string)
work_dir = congfig_path.replace("config", "results").replace(".py", "/")
img_mode = config.test_dataloader.dataset.data_root.name
work_dir = work_dir + img_mode

exp = Inferencer(
    congfig_path=congfig_path,
    inference_root_dir_path=work_dir,
    test_dataloader=config.test_dataloader,
    model=config.model,
    checkpoint_path=config.checkpoint_path,
    metric_list=config.metric_list,
)
exp.inference()
