import argparse
import logging
import time

import torch
import torch.nn.functional as F


def log_result(model_name, avg_time):
    logging.info(f"{model_name}")
    logging.info(f"Time = {avg_time:.4f} s")
    logging.info("-" * 30)


def measure_inference_time(model_name, callable_func, runs=4, warmup=2):
    logging.info(f"正在测试 {model_name} (整图) ...")

    with torch.no_grad():
        for _ in range(warmup):
            callable_func()
            torch.cuda.synchronize()

    total_time = 0.0
    with torch.no_grad():
        for _ in range(runs):
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            callable_func()

            torch.cuda.synchronize()
            total_time += (time.perf_counter() - start_time)

    avg_time = total_time / runs
    log_result(model_name, avg_time)


def patch_inference(model, inputs, patch_size=256, is_mamba=False):
    b, c, h, w = inputs[0].shape
    out = torch.zeros(b, 6, h, w, device=inputs[0].device)


    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            i_end = min(i + patch_size, h)
            j_end = min(j + patch_size, w)

            patches = []
            for inp in inputs:
                patch = inp[:, :, i:i_end, j:j_end]

                pad_h = patch_size - patch.shape[2]
                pad_w = patch_size - patch.shape[3]
                if pad_h > 0 or pad_w > 0:
                    patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='replicate')
                patches.append(patch)


            if is_mamba:
                pred = model(*patches, 'cuda')
            else:
                pred = model(*patches)

            if isinstance(pred, tuple):
                pred = pred[0]


            out[:, :, i:i_end, j:j_end] = pred[:, :, :i_end-i, :j_end-j]

    return out


def main():
    parser = argparse.ArgumentParser(description='Time STFMamba patch inference on a synthetic full scene')
    parser.add_argument('--height', type=int, default=1792)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--patch-size', type=int, default=128)
    parser.add_argument('--runs', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=2)
    args = parser.parse_args()
    if min(args.height, args.width, args.patch_size, args.runs) < 1 or args.warmup < 0:
        parser.error('dimensions and runs must be positive; warmup must be nonnegative')
    if not torch.cuda.is_available():
        parser.error('STFMamba timing requires CUDA and its optional Mamba dependencies')
    from src.model.stfmamba.model import model_STF

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    torch.manual_seed(42)
    inputs = tuple(torch.randn(1, 6, args.height, args.width, device='cuda') for _ in range(3))
    model = model_STF().cuda().eval()
    measure_inference_time(
        'STFMamba', lambda: patch_inference(model, inputs, args.patch_size, is_mamba=True),
        runs=args.runs, warmup=args.warmup,
    )


if __name__ == '__main__':
    main()
