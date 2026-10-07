import os
import numpy as np
import torch
from pathlib import Path


from src.metrics import *
from src.logger import FusionLogger
from tools._metric_io import (
    global_metric_value, load_image_tensor, load_mask_tensor,
    match_image_paths, mean_defined,
)

def batch_offline_evaluation_and_report(methods_dict, gt_dir_path, output_dir, output_log_name, metric_list,
                                        mask_dir_path=None, normalize_scale=10000.0,
                                        gt_layout='HWC', pred_layout='HWC'):
    os.makedirs(output_dir, exist_ok=True)
    gt_dir_path = Path(gt_dir_path)


    txt_logger = FusionLogger(
        logger_name='batch_offline_metric_cal',
        log_level='INFO',
        log_file=os.path.join(output_dir, f'batch_metric_cal_{output_log_name.split(".")[0]}.log')
    )

    txt_output_path = os.path.join(output_dir, output_log_name)
    base_name = os.path.splitext(output_log_name)[0]
    tex_output_path = os.path.join(output_dir, f"{base_name}.tex")

    parsed_results = []
    metric_keys = [metric.__name__ for metric in metric_list]


    gt_img_path_list = sorted([*gt_dir_path.glob('*.tif'), *gt_dir_path.glob('*.tiff')])
    img_num = len(gt_img_path_list)

    if img_num == 0:
        txt_logger.error(f"❌ 错误: 在 GT 目录 {gt_dir_path} 中没有找到任何 .tif 文件！")
        return

    txt_logger.info(f"{'='*60}")
    txt_logger.info(f"🚀 开始批量离线计算与评估，GT图像数量: {img_num}")
    txt_logger.info(f"{'='*60}")


    for method_name, pred_dir_str in methods_dict.items():
        pred_dir_path = Path(pred_dir_str)

        if not pred_dir_path.exists():
            txt_logger.warning(f"⚠️ [跳过] 找不到模型 {method_name} 的预测目录: {pred_dir_path}")
            continue

        pairs = match_image_paths(gt_dir_path, pred_dir_path, mask_dir_path)

        txt_logger.info(f"\n---> 开始评估模型: {method_name}")
        all_results = {metric.__name__: [] for metric in metric_list}
        global_results = {metric.__name__: [] for metric in metric_list}


        for gt_path, pred_path, mask_path in pairs:
            gt_tensor = load_image_tensor(gt_path, normalize_scale, gt_layout)
            pred_tensor = load_image_tensor(pred_path, normalize_scale, pred_layout)
            mask = load_mask_tensor(mask_path, gt_tensor)
            with torch.no_grad():
                for metric in metric_list:
                    val = metric(gt_tensor, pred_tensor, mask=mask)
                    all_results[metric.__name__].append(val.cpu().numpy())
                    global_val = (global_metric_value(metric, gt_tensor, pred_tensor, mask)
                                  if val.numel() > 1 else val.item())
                    global_results[metric.__name__].append(global_val)

        method_metrics_summary = {}
        log_avg_msg = f"✅ {method_name} 评估完成 | 平均结果:"
        txt_line_parts = []

        for metric_name in metric_keys:
            values = np.array(all_results[metric_name])
            mean_val = mean_defined(values, axis=0)


            if mean_val.size > 1:
                avg_global = float(mean_defined(global_results[metric_name]))
                method_metrics_summary[metric_name] = avg_global

                band_str = "/".join([f"{v:.4f}" for v in mean_val])
                log_avg_msg += f" {metric_name}: {avg_global:.4f} ({band_str}),"
                txt_line_parts.append(f"{metric_name}: {avg_global:.4f}")
            else:
                avg_global = float(mean_defined(global_results[metric_name]))
                method_metrics_summary[metric_name] = avg_global

                log_avg_msg += f" {metric_name}: {avg_global:.4f},"
                txt_line_parts.append(f"{metric_name}: {avg_global:.4f}")

        txt_logger.info(log_avg_msg.rstrip(','))


        parsed_results.append({
            "method": method_name,
            "txt_line": f"{method_name}, " + ", ".join(txt_line_parts),
            "metrics": method_metrics_summary
        })


    if not parsed_results:
        txt_logger.error("\n❌ 没有成功评估任何模型，未生成结果文件。")
        return

    with open(txt_output_path, 'w', encoding='utf-8') as f:
        for res in parsed_results:
            f.write(res["txt_line"] + "\n")
    txt_logger.info(f"\n📄 文本结果汇总已保存至: {txt_output_path}")


    with open(tex_output_path, 'w', encoding='utf-8') as f:
        f.write("% 请确保在你的 LaTeX 导言区添加了 \\usepackage{booktabs} 以支持三线表\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("    \\centering\n")
        f.write("    \\caption{Quantitative Evaluation Results}\n")
        f.write("    \\label{tab:results}\n")


        col_format = "l " + " ".join(["c"] * len(metric_keys))
        f.write(f"    \\begin{{tabular}}{{{col_format}}}\n")
        f.write("        \\toprule\n")


        header = "Method & " + " & ".join(metric_keys) + " \\\\\n"
        f.write(f"        {header}")
        f.write("        \\midrule\n")


        for res in parsed_results:
            row_data = [res["method"]]
            for key in metric_keys:

                val = res["metrics"].get(key, None)
                row_data.append(f"{val:.4f}" if val is not None else "-")
            row_str = " & ".join(row_data) + " \\\\\n"
            f.write(f"        {row_str}")


        f.write("        \\bottomrule\n")
        f.write("    \\end{tabular}\n")
        f.write("\\end{table}\n")

    txt_logger.info(f"📊 LaTeX 表格已保存至: {tex_output_path}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate multiple prediction directories')
    parser.add_argument('--gt-dir', required=True)
    parser.add_argument('--method', action='append', required=True, metavar='NAME=PATH')
    parser.add_argument('--output-dir', default='results/evaluation')
    parser.add_argument('--output-log-name', default='evaluation_metrics.txt')
    args = parser.parse_args()
    methods = {}
    for specification in args.method:
        name, separator, directory = specification.partition('=')
        if not separator or not name.strip() or not directory.strip():
            parser.error('each method must have the form NAME=PATH')
        name = name.strip()
        if name in methods:
            parser.error(f'duplicate method name: {name}')
        methods[name] = directory.strip()
    metric_list = [
        RMSE(is_reduce_channel=False), MAE(is_reduce_channel=False),
        PSNRONE(max_value=1.0, is_reduce_channel=False),
        SSIM(data_range=1.0, is_reduce_channel=False), ERGAS(ratio=1.0 / 16.0),
        CC(is_reduce_channel=False), SAM(), UIQI(is_reduce_channel=False),
    ]
    batch_offline_evaluation_and_report(
        methods_dict=methods, gt_dir_path=args.gt_dir, output_dir=args.output_dir,
        output_log_name=args.output_log_name, metric_list=metric_list,
    )
