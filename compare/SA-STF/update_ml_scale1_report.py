import datetime as dt
import json
from pathlib import Path


SA_STF_ROOT = Path(__file__).resolve().parent
COMPARE_ROOT = SA_STF_ROOT.parent


PIXEL_RUNS = [
    ("ML", "AHB 官方预训练迁移 scale1", "full", SA_STF_ROOT / "results/syy_setting-9_ML_official_AHB_weight_scale1/summary_ML_full.json"),
    ("ML", "AHB 官方预训练迁移 scale1", "patch", SA_STF_ROOT / "results/syy_setting-9_ML_official_AHB_weight_scale1/summary_ML_patch.json"),
    ("ML", "自训练 scale1 final", "full", SA_STF_ROOT / "results/syy_setting-9_ML_self_train_scale1/summary_ML_full.json"),
    ("ML", "自训练 scale1 final", "patch", SA_STF_ROOT / "results/syy_setting-9_ML_self_train_scale1/summary_ML_patch.json"),
]

CLASSIFIER_RUNS = [
    ("AHB 官方预训练迁移 scale1", "full", SA_STF_ROOT / "classifier_eval/ML_syy_setting-9_scale1/SA-STF_official_AHB_weight_scale1/full/metrics_summary.json"),
    ("AHB 官方预训练迁移 scale1", "patch", SA_STF_ROOT / "classifier_eval/ML_syy_setting-9_scale1/SA-STF_official_AHB_weight_scale1/patch/metrics_summary.json"),
    ("自训练 scale1 final", "full", SA_STF_ROOT / "classifier_eval/ML_syy_setting-9_scale1/SA-STF_self_train_scale1_final/full/metrics_summary.json"),
    ("自训练 scale1 final", "patch", SA_STF_ROOT / "classifier_eval/ML_syy_setting-9_scale1/SA-STF_self_train_scale1_final/patch/metrics_summary.json"),
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return str(path.relative_to(COMPARE_ROOT))


def fmt(value: float) -> str:
    return f"{value:.6f}"


def fmt_vec(values: list[float]) -> str:
    return "/".join(fmt(value) for value in values)


def pixel_rows() -> list[str]:
    lines = [
        "| 数据集 | 权重类型 | split | 图像数 | PSNRONE | SSIM | RMSE | MAE | CC | SAM | UIQI | ERGAS |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, weight_type, split, path in PIXEL_RUNS:
        rows = load_json(path)
        if len(rows) != 1:
            raise RuntimeError(f"expected one row in {path}, got {len(rows)}")
        row = rows[0]
        lines.append(
            "| {dataset} | {weight_type} | {split} | {num_images} | {psnr} | {ssim} | "
            "{rmse} | {mae} | {cc} | {sam} | {uiqi} | {ergas} |".format(
                dataset=dataset,
                weight_type=weight_type,
                split=split,
                num_images=row["num_images"],
                psnr=fmt(row["PSNRONE"]),
                ssim=fmt(row["SSIM"]),
                rmse=fmt(row["RMSE"]),
                mae=fmt(row["MAE"]),
                cc=fmt(row["CC"]),
                sam=fmt(row["SAM"]),
                uiqi=fmt(row["UIQI"]),
                ergas=fmt(row["ergas"]),
            )
        )
    return lines


def band_rows() -> list[str]:
    rows = [
        row
        for row in load_json(SA_STF_ROOT / "results/sastf_band_metrics.json")
        if row["dataset"] == "ML" and "scale1" in row["weight_type"]
    ]
    lines = [
        "| 权重类型 | split | 图像数 | RMSE(B1-B6) | MAE(B1-B6) | PSNRONE(B1-B6) | SSIM(B1-B6) | CC(B1-B6) | UIQI(B1-B6) | ERGAS | SAM |",
        "|---|---|---:|---|---|---|---|---|---|---:|---:|",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            "| {weight_type} | {split} | {num_images} | {rmse} | {mae} | {psnr} | "
            "{ssim} | {cc} | {uiqi} | {ergas} | {sam} |".format(
                weight_type=row["weight_type"],
                split=row["split"],
                num_images=row["num_images"],
                rmse=fmt_vec(metrics["RMSE"]),
                mae=fmt_vec(metrics["MAE"]),
                psnr=fmt_vec(metrics["PSNRONE"]),
                ssim=fmt_vec(metrics["SSIM"]),
                cc=fmt_vec(metrics["CC"]),
                uiqi=fmt_vec(metrics["UIQI"]),
                ergas=fmt(metrics["ergas"][0]),
                sam=fmt(metrics["SAM"][0]),
            )
        )
    return lines


def classifier_rows() -> list[str]:
    lines = [
        "| 权重类型 | split | items | OA | Kappa | F1 macro | mIoU | Corn F1 | Soybean F1 | Grassland F1 | Forest F1 | Developed F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for weight_type, split, path in CLASSIFIER_RUNS:
        row = load_json(path)
        class_f1 = row["class_f1"]
        lines.append(
            "| {weight_type} | {split} | {items} | {oa} | {kappa} | {f1} | {miou} | "
            "{corn} | {soybean} | {grassland} | {forest} | {developed} |".format(
                weight_type=weight_type,
                split=split,
                items=row["num_items"],
                oa=fmt(row["average_oa"]),
                kappa=fmt(row["average_kappa"]),
                f1=fmt(row["average_f1_macro"]),
                miou=fmt(row["average_miou"]),
                corn=fmt(class_f1["Corn"]),
                soybean=fmt(class_f1["Soybean"]),
                grassland=fmt(class_f1["Grassland"]),
                forest=fmt(class_f1["Forest"]),
                developed=fmt(class_f1["Developed"]),
            )
        )
    return lines


def output_rows() -> list[str]:
    paths = [
        ("训练配置", SA_STF_ROOT / "train_runs/syy_setting-9_ML_scale1/ML/train_config.json"),
        ("训练摘要", SA_STF_ROOT / "train_runs/syy_setting-9_ML_scale1/ML/train_summary.json"),
        ("自训练 full summary", SA_STF_ROOT / "results/syy_setting-9_ML_self_train_scale1/summary_ML_full.json"),
        ("自训练 patch summary", SA_STF_ROOT / "results/syy_setting-9_ML_self_train_scale1/summary_ML_patch.json"),
        ("AHB 迁移 full summary", SA_STF_ROOT / "results/syy_setting-9_ML_official_AHB_weight_scale1/summary_ML_full.json"),
        ("AHB 迁移 patch summary", SA_STF_ROOT / "results/syy_setting-9_ML_official_AHB_weight_scale1/summary_ML_patch.json"),
        ("逐波段 JSON", SA_STF_ROOT / "results/sastf_band_metrics.json"),
        ("逐波段表格", SA_STF_ROOT / "results/sastf_band_metrics_table.md"),
        ("分类评估根目录", SA_STF_ROOT / "classifier_eval/ML_syy_setting-9_scale1"),
    ]
    lines = ["| 项目 | 路径 |", "|---|---|"]
    for name, path in paths:
        lines.append(f"| {name} | `{rel(path)}` |")
    return lines


def main() -> None:
    train_summary = load_json(SA_STF_ROOT / "train_runs/syy_setting-9_ML_scale1/ML/train_summary.json")
    lines = [
        "# SA-STF ML scale1 重训与评估结果",
        "",
        f"- 生成时间：{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- 结论口径：ML 使用 `max_data=1.0` 训练，推理保存使用 `save_scale=1.0`，结果可作为修正后的有效 ML 指标。",
        f"- 训练：200 epoch，batch size {train_summary['batch_size']}，global step {train_summary['global_step']}，device `{train_summary['device']}`。",
        "",
        "## 像素指标",
        "",
        *pixel_rows(),
        "",
        "## 逐波段指标",
        "",
        *band_rows(),
        "",
        "## 分类评估",
        "",
        *classifier_rows(),
        "",
        "## 输出位置",
        "",
        *output_rows(),
        "",
    ]
    output_path = COMPARE_ROOT / "SA-STF_ML_SCALE1_RERUN_RESULTS_ZH.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
