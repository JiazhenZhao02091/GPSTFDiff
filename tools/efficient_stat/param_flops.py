import argparse


def main():
    parser = argparse.ArgumentParser(description='Profile one GPSTFDiff denoiser forward pass')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--without-adamdr', action='store_true')
    args = parser.parse_args()
    if args.image_size < 16 or args.image_size % 16 or args.dim < 8 or args.dim % 8:
        parser.error('image-size must be a multiple of 16; dim must be a multiple of 8')

    import torch
    from thop import profile
    from src.model.GPSTFDiff import PredNoiseNetMKIRA

    torch.manual_seed(42)
    model = PredNoiseNetMKIRA(
        dim=args.dim, channels=6, out_dim=6, dim_mults=(1, 2, 4, 8),
        use_mkira=not args.without_adamdr, mkira_up_indices=(1, 2), mkira_init_alpha=0.0,
    ).eval()
    noisy = torch.randn(1, 6, args.image_size, args.image_size)
    condition = torch.randn(1, 18, args.image_size, args.image_size)
    time = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        macs, params = profile(model, inputs=(noisy, time, condition), verbose=False)
    print(f'Denoiser forward: MACs={macs / 1e9:.4f} G, Params={params / 1e6:.4f} M')


if __name__ == '__main__':
    main()
