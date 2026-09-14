import argparse
import os
import subprocess
import sys
from pathlib import Path


VARIANTS = {
    "unmasked_linear": "config/GPSTFDiff/syy_setting-9/ML/ablation/mask_loss_consine/unmasked_linear.py",
    "masked_linear": "config/GPSTFDiff/syy_setting-9/ML/ablation/mask_loss_consine/masked_linear.py",
    "unmasked_cosine": "config/GPSTFDiff/syy_setting-9/ML/ablation/mask_loss_consine/unmasked_cosine.py",
    "masked_cosine": "config/GPSTFDiff/syy_setting-9/ML/ablation/mask_loss_consine/masked_cosine.py",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the GPSTFDiff 2x2 mask-loss and noise-schedule ablation on ML."
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("compare/abl/mask_loss_consine/ML"),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS.keys()),
        choices=list(VARIANTS.keys()),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 3407, 2024])
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument("--val_interval", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    for variant in args.variants:
        config_path = VARIANTS[variant]
        for seed in args.seeds:
            work_dir = args.output_root / variant / f"seed_{seed}"
            env = os.environ.copy()
            env["GPSTFDIFF_SEED"] = str(seed)
            if args.max_epoch is not None:
                env["GPSTFDIFF_MAX_EPOCH"] = str(args.max_epoch)
            if args.val_interval is not None:
                env["GPSTFDIFF_VAL_INTERVAL"] = str(args.val_interval)
            if args.save_interval is not None:
                env["GPSTFDIFF_SAVE_INTERVAL"] = str(args.save_interval)

            cmd = [
                sys.executable,
                "tools/train/train_GPSTFDiff.py",
                "--congfig_path",
                config_path,
                "--work_dir",
                str(work_dir),
                "--seed",
                str(seed),
            ]
            print(" ".join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, cwd=repo_root, env=env, check=True)


if __name__ == "__main__":
    main()
