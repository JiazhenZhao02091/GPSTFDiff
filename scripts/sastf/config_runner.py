import argparse
import importlib.util
import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SA_STF_ROOT = REPO_ROOT / "compare" / "SA-STF"


def load_config(config_path):
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location("sastf_runtime_config", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def _get(config, name, default=None):
    return getattr(config, name, default)


def _append_arg(argv, name, value):
    if value is None:
        return
    cli_name = "--" + name.replace("_", "-")
    if isinstance(value, bool):
        if value:
            argv.append(cli_name)
        return
    argv.extend([cli_name, str(value)])


def build_train_argv(config):
    argv = [str(SA_STF_ROOT / "train_syy_setting9.py")]
    for name in [
        "dataset",
        "data_root",
        "output_root",
        "device",
        "epochs",
        "batch_size",
        "num_workers",
        "timesteps",
        "sampling_steps",
        "ddim_eta",
        "inner_channel",
        "lr",
        "grad_clip",
        "save_every",
        "max_data",
        "seed",
        "resume",
        "max_steps",
        "perceptual",
    ]:
        _append_arg(argv, name, _get(config, name))
    return argv


def build_inference_argv(config):
    argv = [str(SA_STF_ROOT / "run_syy_setting9.py")]
    for name in [
        "dataset",
        "split",
        "data_root",
        "result_root",
        "weights_dir",
        "weight_path",
        "download_weights",
        "device",
        "limit",
        "epoch",
        "timesteps",
        "sampling_steps",
        "ddim_eta",
        "inner_channel",
        "patch_size",
        "max_data",
        "save_scale",
        "seed",
    ]:
        _append_arg(argv, name, _get(config, name))
    return argv


def run_sa_stf(argv):
    script_path = Path(argv[0])
    old_argv = sys.argv[:]
    old_path = sys.path[:]
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        if str(SA_STF_ROOT) not in sys.path:
            sys.path.insert(0, str(SA_STF_ROOT))
        sys.argv = argv
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path


def parse_config_arg(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--congfig_path", "--config_path", dest="config_path", required=True)
    return parser.parse_args()
