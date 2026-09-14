# python tools/train/train_sastf.py --congfig_path config/sastf/syy_setting-9/CIA/train.py

import os
import sys


PROJECT_ROOT = os.getcwd()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["OMP_NUM_THREADS"] = "1"

from scripts.sastf.config_runner import build_train_argv, load_config, parse_config_arg, run_sa_stf


def main():
    args = parse_config_arg("Train SA-STF from a repository config file.")
    config, _ = load_config(args.config_path)
    run_sa_stf(build_train_argv(config))


if __name__ == "__main__":
    main()
