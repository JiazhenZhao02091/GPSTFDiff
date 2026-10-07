import argparse
import time


def main():
    parser = argparse.ArgumentParser(description='Time GPSTFDiff progressive sampling on synthetic inputs')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--sampling-steps', type=int, default=100)
    parser.add_argument('--runs', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--without-adamdr', action='store_true')
    args = parser.parse_args()
    if args.image_size < 64 or args.image_size % 64 or args.dim < 8 or args.dim % 8:
        parser.error('image-size must be a multiple of 64; dim must be a multiple of 8')
    if args.runs < 1 or args.warmup < 0 or not 20 <= args.sampling_steps <= 1000 or args.sampling_steps % 20:
        parser.error('runs must be positive; warmup nonnegative; sampling-steps a multiple of 20 in [20, 1000]')

    import torch
    from src.model.GPSTFDiff import PredNoiseNetMKIRA
    from src.model.GPSTFDiff.diffusion_inferencer import Diffusion

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; select --device cpu')
    torch.manual_seed(42)
    def denoiser():
        return PredNoiseNetMKIRA(
            dim=args.dim, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8),
            use_mkira=not args.without_adamdr, mkira_up_indices=(1, 2), mkira_init_alpha=0.0,
        )
    model = Diffusion(
        model=denoiser(), model_x3=denoiser(), model_x2_x3=denoiser(),
        model_x1_x2_x3=denoiser(), image_size=args.image_size,
        sampling_timesteps=args.sampling_steps,
    ).to(device).eval()
    inputs = tuple(torch.randn(1, 6, args.image_size, args.image_size, device=device) for _ in range(3))
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elif device.type == 'mps':
            torch.mps.synchronize()
    with torch.no_grad():
        for _ in range(args.warmup):
            model.sample(*inputs)
        synchronize()
        elapsed = 0.0
        for _ in range(args.runs):
            synchronize()
            start = time.perf_counter()
            output = model.sample(*inputs)
            synchronize()
            elapsed += time.perf_counter() - start
    if output.shape != inputs[0].shape:
        raise RuntimeError(f'Unexpected sampled output shape: {tuple(output.shape)}')
    print(f'Synthetic progressive sampling: {elapsed / args.runs:.4f} s/image on {device}')


if __name__ == '__main__':
    main()
