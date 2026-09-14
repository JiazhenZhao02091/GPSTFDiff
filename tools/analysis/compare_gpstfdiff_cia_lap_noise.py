import argparse
import re
from pathlib import Path


METRIC_ORDER = [
    "loss",
    "rmse",
    "mae",
    "psnr",
    "ssim",
    "ergas",
    "cc",
    "sam",
    "uiqi",
]


def parse_last_metric_line(log_path):
    metric_line = None
    for line in Path(log_path).read_text().splitlines():
        if "val epoch:" in line and ", rmse:" in line.lower():
            metric_line = line

    if metric_line is None:
        raise ValueError(f"no final metric line found in {log_path}")

    metrics = {}
    for name, value in re.findall(r"([A-Za-z_]+):\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", metric_line):
        metrics[name.lower()] = float(value)
    return metrics, metric_line


def fmt(value):
    return f"{value:.6f}"


def main():
    parser = argparse.ArgumentParser(
        description="Compare current GPSTFDiff CIA baseline metrics with lap-noise single-model metrics."
    )
    parser.add_argument("--baseline_log", required=True)
    parser.add_argument("--lap_noise_log", required=True)
    parser.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    args = parser.parse_args()

    baseline, baseline_line = parse_last_metric_line(args.baseline_log)
    lap_noise, lap_noise_line = parse_last_metric_line(args.lap_noise_log)

    if args.format == "csv":
        print("metric,baseline,lap_noise,delta")
        for metric in METRIC_ORDER:
            if metric in baseline and metric in lap_noise:
                print(
                    f"{metric},{fmt(baseline[metric])},{fmt(lap_noise[metric])},"
                    f"{fmt(lap_noise[metric] - baseline[metric])}"
                )
        return

    print("| metric | baseline | lap_noise_single_model | delta |")
    print("| --- | ---: | ---: | ---: |")
    for metric in METRIC_ORDER:
        if metric in baseline and metric in lap_noise:
            print(
                f"| {metric} | {fmt(baseline[metric])} | {fmt(lap_noise[metric])} | "
                f"{fmt(lap_noise[metric] - baseline[metric])} |"
            )
    print()
    print(f"baseline: {baseline_line}")
    print(f"lap_noise: {lap_noise_line}")


if __name__ == "__main__":
    main()
