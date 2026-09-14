#!/usr/bin/env python3
"""
功能:
    运行并汇总 GPSTFDiff stage-boundary 敏感性实验，按日志中的指标生成论文表格和热力图。

参数:
    --mode (str): 选择 paper、heatmap 或 all 实验集合。
    --summarize-only (flag): 只解析已有日志并生成汇总文件，不启动推理。
    --skip-existing / --no-skip-existing (flag): 控制是否跳过已有最终指标的配置。
    --gpu (str): 写入 CUDA_VISIBLE_DEVICES 的 GPU 编号。
    --data-root (str): 覆盖 CIA 测试数据根目录。
    --result-split (str): 指定 full 或 patch 等结果子目录。
    --ckpt-x3 / --ckpt-x2-x3 / --ckpt-x1-x2-x3 (str): 覆盖不同阶段的 checkpoint 路径。
    --heatmap-metric (str): 用于热力图着色的指标。
    --final-k1 / --final-k2 (int): 热力图中标记的最终 stage 边界。

返回:
    None。main 函数不返回数据对象。

输出:
    必要时启动推理；在 results/stage_boundary_sweep 下写出 CSV、JSON、LaTeX、SVG/PDF/PNG 汇总文件。
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = (
    REPO_ROOT
    / "config"
    / "GPSTFDiff"
    / "syy_setting-9"
    / "CIA"
    / "ablation"
    / "stage_boundary"
)
SUMMARY_DIR = REPO_ROOT / "results" / "stage_boundary_sweep"
METRICS = ["RMSE", "PSNR", "SSIM", "ERGAS", "CC", "SAM"]


def import_manifest():
    sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module(
        "config.GPSTFDiff.syy_setting-9.CIA.ablation.stage_boundary.manifest"
    )


def config_path(t1: int, t2: int) -> Path:
    return CONFIG_DIR / f"inference_T{t1:03d}_T{t2:03d}.py"


def result_dir_for_config(path: Path, result_split: str = "full") -> Path:
    rel_no_ext = path.relative_to(REPO_ROOT).with_suffix("")
    return REPO_ROOT / str(rel_no_ext).replace("config", "results", 1) / result_split


def log_path_for_config(path: Path, result_split: str = "full") -> Path:
    return result_dir_for_config(path, result_split) / "txt_logs" / "log.log"


def format_float(value, digits=4):
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def parse_metrics_from_log(log_path: Path):
    if not log_path.exists():
        return None
    final_line = None
    time_line = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "总耗时:" in line:
            time_line = line
        if "val epoch:" in line and all(f"{metric}:".lower() in line.lower() for metric in METRICS):
            final_line = line
    if final_line is None:
        return None

    row = {}
    for key in ["loss"] + METRICS:
        match = re.search(rf"{key}:\s*([0-9.eE+-]+)", final_line, re.IGNORECASE)
        if match:
            row[key] = float(match.group(1))
    if time_line is not None:
        match = re.search(r"总耗时:\s*([0-9.eE+-]+)", time_line)
        if match:
            row["Time"] = float(match.group(1))
    return row


def ensure_config_for_experiment(exp):
    path = config_path(exp["t1"], exp["t2"])
    if path.exists():
        return path
    path.write_text(
        "from .base import export\n\n"
        f"export(globals(), t1={exp['t1']}, t2={exp['t2']})\n",
        encoding="utf-8",
    )
    return path


def collect_rows(experiments, result_split="full"):
    rows = []
    manifest = import_manifest()
    for exp in experiments:
        path = ensure_config_for_experiment(exp)
        metrics = parse_metrics_from_log(log_path_for_config(path, result_split)) or {}
        t1 = int(exp["t1"])
        t2 = int(exp["t2"])
        s = int(manifest.SAMPLING_TIMESTEPS)
        total = int(manifest.NUM_TRAIN_TIMESTEPS)
        k1 = int(round(t1 / total * s))
        k2 = int(round(t2 / total * s))
        rows.append(
            dict(
                Setting=exp.get("name", f"K1_{k1:02d}_K2_{k2:02d}"),
                T1=t1,
                T2=t2,
                K1=k1,
                K2=k2,
                S3=s - k2,
                S2=k2 - k1,
                S1=k1,
                config=str(path.relative_to(REPO_ROOT)),
                log=str(log_path_for_config(path, result_split).relative_to(REPO_ROOT)),
                completed=bool(metrics),
                **metrics,
            )
        )
    return rows


def run_experiments(experiments, args):
    env = os.environ.copy()
    if args.data_root:
        env["GPSTFDIFF_STAGE_DATA_ROOT"] = args.data_root
    if args.ckpt_x3:
        env["GPSTFDIFF_STAGE_CKPT_X3"] = args.ckpt_x3
    if args.ckpt_x2_x3:
        env["GPSTFDIFF_STAGE_CKPT_X2_X3"] = args.ckpt_x2_x3
    if args.ckpt_x1_x2_x3:
        env["GPSTFDIFF_STAGE_CKPT_X1_X2_X3"] = args.ckpt_x1_x2_x3
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    result_split = get_result_split(args)
    for index, exp in enumerate(experiments, start=1):
        path = ensure_config_for_experiment(exp)
        log_path = log_path_for_config(path, result_split)
        if args.skip_existing and parse_metrics_from_log(log_path):
            print(f"[{index}/{len(experiments)}] skip completed {path}")
            continue

        cmd = [
            sys.executable,
            "tools/inference/test_GPSTFDiff_lap.py",
            "--congfig_path",
            str(path.relative_to(REPO_ROOT)),
        ]
        print(f"[{index}/{len(experiments)}] run {' '.join(cmd)}")
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def write_csv(rows, out_path):
    fields = [
        "Setting",
        "T1",
        "T2",
        "K1",
        "K2",
        "S3",
        "S2",
        "S1",
        *METRICS,
        "Time",
        "completed",
        "config",
        "log",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def latex_escape(value):
    return str(value).replace("_", r"\_")


def best_by_metric(rows, metric):
    valid = [row for row in rows if isinstance(row.get(metric), float)]
    if not valid:
        return None
    reverse = metric in {"PSNR", "SSIM", "CC"}
    return sorted(valid, key=lambda row: row[metric], reverse=reverse)[0]


def write_latex(rows, out_path, caption, label):
    best_rows = {metric: best_by_metric(rows, metric) for metric in METRICS}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{c|cc|ccc|ccccccc}",
        r"\hline",
        r"Setting & $T_1$ & $T_2$ & $K_1$ & $K_2$ & S3/S2/S1 & RMSE$\downarrow$ & PSNR$\uparrow$ & SSIM$\uparrow$ & ERGAS$\downarrow$ & CC$\uparrow$ & SAM$\downarrow$ & Time (s) \\",
        r"\hline",
    ]
    for row in rows:
        cells = [
            latex_escape(row["Setting"]),
            row["T1"],
            row["T2"],
            row["K1"],
            row["K2"],
            f"{row['S3']}/{row['S2']}/{row['S1']}",
        ]
        for metric in METRICS:
            value = format_float(row.get(metric))
            if best_rows.get(metric) is row and value != "--":
                value = r"\textbf{" + value + r"}"
            cells.append(value)
        cells.append(format_float(row.get("Time"), digits=2))
        lines.append(" & ".join(map(str, cells)) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def value_to_color(value, min_value, max_value):
    if value is None or math.isnan(value):
        return "#f3f4f6"
    if max_value <= min_value:
        ratio = 0.5
    else:
        ratio = (value - min_value) / (max_value - min_value)
    ratio = max(0.0, min(1.0, ratio))
    # Low values are blue-green, high values are warm red.
    r = round(59 + ratio * (213 - 59))
    g = round(130 + ratio * (94 - 130))
    b = round(246 + ratio * (87 - 246))
    return f"#{r:02x}{g:02x}{b:02x}"


def write_svg_heatmap(rows, out_path, metric, final_k1, final_k2):
    valid_rows = [row for row in rows if isinstance(row.get(metric), float)]
    k1_values = sorted({int(row["K1"]) for row in rows})
    k2_values = sorted({int(row["K2"]) for row in rows})
    values = [row[metric] for row in valid_rows]
    min_value = min(values) if values else 0.0
    max_value = max(values) if values else 1.0

    cell = 46
    left = 72
    top = 48
    right = 112
    bottom = 72
    width = left + len(k1_values) * cell + right
    height = top + len(k2_values) * cell + bottom
    lookup = {(int(row["K1"]), int(row["K2"])): row for row in rows}

    def text(x, y, content, size=12, anchor="middle", weight="400", color="#111827"):
        return (
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'font-family="Arial, sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}">{content}</text>'
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(width / 2, 24, f"Stage-boundary sensitivity ({metric})", 15, weight="700"),
        text(width / 2, height - 20, r"K1: switch to finest stage", 12),
        f'<g transform="translate(18 {height / 2}) rotate(-90)">'
        + text(0, 0, r"K2: switch to intermediate stage", 12)
        + "</g>",
    ]

    for col, k1 in enumerate(k1_values):
        x = left + col * cell
        parts.append(text(x + cell / 2, top + len(k2_values) * cell + 22, str(k1), 12))
    for row_index, k2 in enumerate(reversed(k2_values)):
        y = top + row_index * cell
        parts.append(text(left - 16, y + cell / 2 + 4, str(k2), 12, anchor="end"))
        for col, k1 in enumerate(k1_values):
            x = left + col * cell
            row = lookup.get((k1, k2))
            value = row.get(metric) if row else None
            color = value_to_color(value, min_value, max_value)
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="{color}" stroke="#ffffff" stroke-width="2"/>'
            )
            label = format_float(value, 4) if value is not None else "--"
            parts.append(text(x + cell / 2, y + cell / 2 + 4, label, 10, color="#111827"))
            if k1 == final_k1 and k2 == final_k2:
                parts.append(
                    f'<rect x="{x + 3}" y="{y + 3}" width="{cell - 6}" '
                    f'height="{cell - 6}" fill="none" stroke="#111827" stroke-width="3"/>'
                )
                parts.append(text(x + cell - 7, y + 13, "*", 16, weight="700"))

    legend_x = left + len(k1_values) * cell + 26
    legend_y = top
    legend_h = len(k2_values) * cell
    for i in range(80):
        ratio = i / 79
        color = value_to_color(min_value + ratio * (max_value - min_value), min_value, max_value)
        y = legend_y + (79 - i) / 80 * legend_h
        parts.append(
            f'<rect x="{legend_x}" y="{y}" width="16" height="{legend_h / 80 + 1}" fill="{color}"/>'
        )
    parts.append(text(legend_x + 24, legend_y + 4, format_float(max_value), 10, anchor="start"))
    parts.append(text(legend_x + 24, legend_y + legend_h, format_float(min_value), 10, anchor="start"))
    parts.append(text(legend_x, legend_y + legend_h + 26, "* final", 11, anchor="start", weight="700"))
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_matplotlib_heatmap_if_available(rows, out_path, metric, final_k1, final_k2):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return False

    k1_values = sorted({int(row["K1"]) for row in rows})
    k2_values = sorted({int(row["K2"]) for row in rows})
    lookup = {(int(row["K1"]), int(row["K2"])): row for row in rows}
    grid = np.full((len(k2_values), len(k1_values)), np.nan)
    for y, k2 in enumerate(k2_values):
        for x, k1 in enumerate(k1_values):
            value = lookup.get((k1, k2), {}).get(metric)
            if isinstance(value, float):
                grid[y, x] = value

    fig, ax = plt.subplots(figsize=(5.0, 3.8), dpi=300)
    im = ax.imshow(grid, origin="lower", cmap="viridis_r")
    ax.set_xticks(range(len(k1_values)), k1_values)
    ax.set_yticks(range(len(k2_values)), k2_values)
    ax.set_xlabel(r"$K_1$")
    ax.set_ylabel(r"$K_2$")
    ax.set_title(f"Stage-boundary sensitivity ({metric})")
    for y, k2 in enumerate(k2_values):
        for x, k1 in enumerate(k1_values):
            if math.isfinite(grid[y, x]):
                ax.text(x, y, f"{grid[y, x]:.4f}", ha="center", va="center", fontsize=6)
    if final_k1 in k1_values and final_k2 in k2_values:
        ax.scatter(
            [k1_values.index(final_k1)],
            [k2_values.index(final_k2)],
            marker="*",
            s=120,
            c="white",
            edgecolors="black",
            linewidths=0.8,
            label="Final",
        )
        ax.legend(loc="upper right", fontsize=7, frameon=True)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=metric)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def write_outputs(paper_rows, heatmap_rows, args):
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(paper_rows, SUMMARY_DIR / "stage_boundary_table.csv")
    write_csv(heatmap_rows, SUMMARY_DIR / "stage_boundary_heatmap.csv")
    (SUMMARY_DIR / "stage_boundary_table.json").write_text(
        json.dumps(paper_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_latex(
        paper_rows,
        SUMMARY_DIR / "stage_boundary_table.tex",
        "Sensitivity of GPSTFDiff to progressive stage switching steps on CIA.",
        "tab:stage_boundary_sensitivity",
    )
    final_k1 = args.final_k1
    final_k2 = args.final_k2
    write_svg_heatmap(
        heatmap_rows,
        SUMMARY_DIR / f"stage_boundary_heatmap_{args.heatmap_metric}.svg",
        args.heatmap_metric,
        final_k1,
        final_k2,
    )
    wrote_pdf = write_matplotlib_heatmap_if_available(
        heatmap_rows,
        SUMMARY_DIR / f"stage_boundary_heatmap_{args.heatmap_metric}.pdf",
        args.heatmap_metric,
        final_k1,
        final_k2,
    )
    if wrote_pdf:
        write_matplotlib_heatmap_if_available(
            heatmap_rows,
            SUMMARY_DIR / f"stage_boundary_heatmap_{args.heatmap_metric}.png",
            args.heatmap_metric,
            final_k1,
            final_k2,
        )
    print(f"Wrote summary artifacts to {SUMMARY_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["paper", "heatmap", "all"],
        default="all",
        help="Which experiment set to run and summarize.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Do not launch inference; only parse existing logs and write artifacts.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip configs whose final metrics are already present in log.log.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Force rerunning configs even when log.log already contains metrics.",
    )
    parser.add_argument("--gpu", help="Value for CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--data-root", help="Override CIA test/full data root.")
    parser.add_argument(
        "--result-split",
        help="Result subdirectory produced by the inference entrypoint, usually full or patch. Defaults to the --data-root basename.",
    )
    parser.add_argument("--ckpt-x3", help="Override x3 checkpoint path.")
    parser.add_argument("--ckpt-x2-x3", help="Override x2+x3 checkpoint path.")
    parser.add_argument("--ckpt-x1-x2-x3", help="Override x1+x2+x3 checkpoint path.")
    parser.add_argument(
        "--heatmap-metric",
        choices=METRICS,
        default="RMSE",
        help="Metric used as the heatmap color value.",
    )
    parser.add_argument("--final-k1", type=int, default=10)
    parser.add_argument("--final-k2", type=int, default=30)
    return parser.parse_args()


# python tools/analysis/run_stage_boundary_sweep.py \
#   --mode all \
#   --gpu 4 \
#   --data-root data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/test/patch \
#   --ckpt-x3  /home/zhaojiazhen/workspace/STF/STF/checkpoints/best_model_epoch_149.pth \
#   --ckpt-x2-x3 /home/zhaojiazhen/workspace/STF/STF/checkpoints/best_model_epoch_149.pth  \
#   --ckpt-x1-x2-x3 /home/zhaojiazhen/workspace/STF/STF/checkpoints/best_model_epoch_149.pth 



def get_result_split(args):
    if args.result_split:
        return args.result_split
    if args.data_root:
        return Path(args.data_root).name
    return "full"


def main():
    args = parse_args()
    manifest = import_manifest()
    paper_experiments = manifest.PAPER_SWEEP
    heatmap_experiments = manifest.build_heatmap_sweep()
    if args.mode == "paper":
        to_run = paper_experiments
    elif args.mode == "heatmap":
        to_run = heatmap_experiments
    else:
        to_run = manifest.all_experiments()

    if not args.summarize_only:
        run_experiments(to_run, args)

    result_split = get_result_split(args)
    paper_rows = collect_rows(paper_experiments, result_split=result_split)
    heatmap_rows = collect_rows(heatmap_experiments, result_split=result_split)
    write_outputs(paper_rows, heatmap_rows, args)


if __name__ == "__main__":
    main()
