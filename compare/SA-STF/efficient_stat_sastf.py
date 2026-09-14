import argparse
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import torch
from thop import profile


SA_STF_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SA_STF_ROOT.parents[1]


def load_sa_stf_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_model_classes():
    diffusion_pkg = types.ModuleType("Diffusion")
    diffusion_pkg.__path__ = [str(SA_STF_ROOT / "Diffusion")]
    sys.modules["Diffusion"] = diffusion_pkg
    load_sa_stf_module("Diffusion.loss_perceptual", SA_STF_ROOT / "Diffusion" / "loss_perceptual.py")
    model_module = load_sa_stf_module("Diffusion.Model", SA_STF_ROOT / "Diffusion" / "Model.py")
    diffusion_res_module = load_sa_stf_module(
        "Diffusion.diffusion_res", SA_STF_ROOT / "Diffusion" / "diffusion_res.py"
    )

    class DummyPerceptualLoss(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, inputs, targets):
            return torch.zeros((), device=inputs.device)

    diffusion_res_module.PerceptualLoss = DummyPerceptualLoss
    diffusion_res_module.tqdm = lambda iterable, *args, **kwargs: iterable
    return model_module.UNet, diffusion_res_module.ResidualDiffusion


def build_unet(unet_cls):
    model = unet_cls(
        in_channel=6,
        out_channel=6,
        inner_channel=64,
        norm_groups=32,
        channel_mults=[1, 2, 4, 8, 8],
        dropout=0,
    )
    model.eval()
    return model


def random_unet_inputs(device, patch_size):
    shape = (1, 6, patch_size, patch_size)
    xt = torch.randn(shape, device=device)
    lrsr = torch.randn(shape, device=device)
    ref0 = torch.randn(shape, device=device)
    refsr0 = torch.randn(shape, device=device)
    ref1 = torch.randn(shape, device=device)
    refsr1 = torch.randn(shape, device=device)
    t0 = torch.rand(1, 1, device=device) * 100
    t1 = torch.rand(1, 1, device=device) * 100
    return xt, [t0, t1], lrsr, refsr0, ref0, refsr1, ref1


@torch.no_grad()
def measure_cuda(callable_func, runs, warmup):
    for _ in range(warmup):
        callable_func()
        torch.cuda.synchronize()

    elapsed = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        callable_func()
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - start)
    return {
        "runs": runs,
        "warmup": warmup,
        "times_s": elapsed,
        "avg_s": sum(elapsed) / len(elapsed),
        "min_s": min(elapsed),
        "max_s": max(elapsed),
    }


def patch_indices(size: int, patch_size: int, patch_stride: int):
    end = (size - patch_stride) // patch_stride * patch_stride
    indices = [i for i in range(0, end, patch_stride)]
    if (size - patch_stride) % patch_stride != 0:
        indices.append(size - patch_size)
    return sorted(set(max(0, i) for i in indices))


def full_image_shapes_and_patch_counts(patch_size: int):
    try:
        import tifffile as tiff
    except Exception:
        return []

    rows = []
    for dataset in ("CIA", "LGC", "ML"):
        img_dir = (
            REPO_ROOT
            / "data"
            / "spatio_temporal_fusion"
            / dataset
            / "private_data"
            / "syy_setting-9"
            / "test"
            / "full"
            / "Landsat_02"
        )
        if not img_dir.exists():
            continue
        for path in sorted(img_dir.glob("*.tif")):
            arr = tiff.imread(path)
            height, width = arr.shape[:2]
            stride = patch_size // 2
            h_count = len(patch_indices(height, patch_size, stride))
            w_count = len(patch_indices(width, patch_size, stride))
            rows.append(
                {
                    "dataset": dataset,
                    "image": path.name,
                    "height": height,
                    "width": width,
                    "patch_size": patch_size,
                    "patch_stride": stride,
                    "patch_count": h_count * w_count,
                    "h_patch_count": h_count,
                    "w_patch_count": w_count,
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description="SA-STF efficiency statistics.")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--sample-runs", type=int, default=3)
    parser.add_argument("--sample-warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, default=SA_STF_ROOT / "efficient_stat_sastf.json")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.backends.cudnn.benchmark = True

    unet_cls, diffusion_cls = load_model_classes()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    unet_cpu = build_unet(unet_cls)
    flops, thop_params = profile(
        unet_cpu,
        inputs=random_unet_inputs(torch.device("cpu"), args.patch_size),
        verbose=False,
    )
    params = sum(p.numel() for p in unet_cpu.parameters())

    result = {
        "model": "SA-STF",
        "device": str(device),
        "torch": torch.__version__,
        "patch_size": args.patch_size,
        "timesteps": args.timesteps,
        "sampling_steps": args.sampling_steps,
        "params": params,
        "params_m": params / 1000**2,
        "thop_params": thop_params,
        "thop_params_m": thop_params / 1000**2,
        "unet_forward_flops": flops,
        "unet_forward_flops_g": flops / 1000**3,
        "sample_flops_estimate": flops * args.sampling_steps,
        "sample_flops_estimate_g": flops * args.sampling_steps / 1000**3,
        "full_images": full_image_shapes_and_patch_counts(args.patch_size),
    }

    if device.type == "cuda":
        unet = build_unet(unet_cls).to(device)
        unet_inputs = random_unet_inputs(device, args.patch_size)
        result["unet_forward_time"] = measure_cuda(
            lambda: unet(*unet_inputs), runs=args.runs, warmup=args.warmup
        )

        sampler = diffusion_cls(
            model_x0=unet,
            total_epoch=200,
            timesteps=args.timesteps,
            sampling_steps=args.sampling_steps,
            ddim_sampling_eta=0.0,
            device=device,
        ).to(device)
        sampler.eval()
        _, _, lrsr, refsr0, ref0, refsr1, ref1 = unet_inputs
        result["sample_time"] = measure_cuda(
            lambda: sampler.sample(lrsr, ref0, refsr0, ref1, refsr1),
            runs=args.sample_runs,
            warmup=args.sample_warmup,
        )
    else:
        result["unet_forward_time"] = None
        result["sample_time"] = None

    if result["sample_time"] is not None:
        patch_time = result["sample_time"]["avg_s"]
        for row in result["full_images"]:
            row["estimated_full_time_s"] = row["patch_count"] * patch_time

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
