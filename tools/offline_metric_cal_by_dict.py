import os
import numpy as np
import torch
import tifffile as tiff
from pathlib import Path

# 导入你自定义的库 (确保在你的环境中可用)
from src.metrics import *
from src.logger import FusionLogger

def batch_offline_evaluation_and_report(methods_dict, gt_dir_path, output_dir, output_log_name, metric_list):
    """
    批量进行离线指标计算，并自动生成汇总 TXT 和 LaTeX 表格。
    
    参数:
        methods_dict (dict): { '方法名': '预测结果所在目录路径' }
        gt_dir_path (str/Path): Ground Truth 真值图像所在目录
        output_dir (str/Path): 最终结果和表格的保存目录
        output_log_name (str): 汇总 txt 的名称 (例如 'compiled_metrics.txt')
        metric_list (list): 实例化的指标对象列表
    """
    os.makedirs(output_dir, exist_ok=True)
    gt_dir_path = Path(gt_dir_path)
    
    # 初始化全局日志记录器
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
    
    # 获取 GT 列表
    gt_img_path_list = sorted(list(gt_dir_path.glob('*.tif')))
    img_num = len(gt_img_path_list)
    
    if img_num == 0:
        txt_logger.error(f"❌ 错误: 在 GT 目录 {gt_dir_path} 中没有找到任何 .tif 文件！")
        return

    txt_logger.info(f"{'='*60}")
    txt_logger.info(f"🚀 开始批量离线计算与评估，GT图像数量: {img_num}")
    txt_logger.info(f"{'='*60}")
    
    # 1. 遍历每个方法进行计算
    for method_name, pred_dir_str in methods_dict.items():
        pred_dir_path = Path(pred_dir_str)
        
        if not pred_dir_path.exists():
            txt_logger.warning(f"⚠️ [跳过] 找不到模型 {method_name} 的预测目录: {pred_dir_path}")
            continue
            
        pred_img_path_list = sorted(list(pred_dir_path.glob('*.tif')))
        
        if len(pred_img_path_list) != img_num:
            txt_logger.warning(f"⚠️ [跳过] {method_name} 的预测图片数量 ({len(pred_img_path_list)}) 与 GT ({img_num}) 不匹配！")
            continue
            
        txt_logger.info(f"\n---> 开始评估模型: {method_name}")
        all_results = {metric.__name__: [] for metric in metric_list}
        
        # 逐图计算
        for i in range(img_num):
            gt_img = tiff.imread(str(gt_img_path_list[i]))
            pred_img = tiff.imread(str(pred_img_path_list[i]))
            
            # 预处理：归一化并转为 Tensor (1, C, H, W)
            gt_img = (gt_img.astype(np.float32) / 10000.0)
            pred_img = (pred_img.astype(np.float32) / 10000.0)
            
            gt_tensor = torch.from_numpy(gt_img.transpose(2, 0, 1)).unsqueeze(0)
            pred_tensor = torch.from_numpy(pred_img.transpose(2, 0, 1)).unsqueeze(0)
            
            # 计算各项指标
            for metric in metric_list:
                val = metric(gt_tensor, pred_tensor)
                if isinstance(val, torch.Tensor):
                    val_numpy = val.cpu().numpy()
                else:
                    val_numpy = np.array(val)
                all_results[metric.__name__].append(val_numpy)
                
        # 2. 计算该方法在所有图片上的平均值
        method_metrics_summary = {}
        log_avg_msg = f"✅ {method_name} 评估完成 | 平均结果:"
        txt_line_parts = []
        
        for metric_name in metric_keys:
            values = np.array(all_results[metric_name])
            mean_val = np.mean(values, axis=0) # 对图片数量维度求均值
            
            # 处理多波段情况 (Vector) vs 单一标量 (Scalar)
            if mean_val.size > 1:
                avg_global = float(np.mean(mean_val)) # 提取全局平均值用于制表
                method_metrics_summary[metric_name] = avg_global
                
                band_str = "/".join([f"{v:.4f}" for v in mean_val])
                log_avg_msg += f" {metric_name}: {avg_global:.4f} ({band_str}),"
                txt_line_parts.append(f"{metric_name}: {avg_global:.4f}")
            else:
                avg_global = float(mean_val)
                method_metrics_summary[metric_name] = avg_global
                
                log_avg_msg += f" {metric_name}: {avg_global:.4f},"
                txt_line_parts.append(f"{metric_name}: {avg_global:.4f}")
                
        txt_logger.info(log_avg_msg.rstrip(','))
        
        # 存储制表所需的数据
        parsed_results.append({
            "method": method_name,
            "txt_line": f"{method_name}, " + ", ".join(txt_line_parts),
            "metrics": method_metrics_summary
        })

    # 3. 写入 TXT 结果汇总
    if not parsed_results:
        txt_logger.error("\n❌ 没有成功评估任何模型，未生成结果文件。")
        return
        
    with open(txt_output_path, 'w', encoding='utf-8') as f:
        for res in parsed_results:
            f.write(res["txt_line"] + "\n")
    txt_logger.info(f"\n📄 文本结果汇总已保存至: {txt_output_path}")
    
    # 4. 生成并写入 LaTeX 表格代码
    with open(tex_output_path, 'w', encoding='utf-8') as f:
        f.write("% 请确保在你的 LaTeX 导言区添加了 \\usepackage{booktabs} 以支持三线表\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("    \\centering\n")
        f.write("    \\caption{Quantitative Evaluation Results}\n")
        f.write("    \\label{tab:results}\n")
        
        # 动态生成列格式，例如: l c c c c ...
        col_format = "l " + " ".join(["c"] * len(metric_keys))
        f.write(f"    \\begin{{tabular}}{{{col_format}}}\n")
        f.write("        \\toprule\n")
        
        # 写入表头 (指标名)
        header = "Method & " + " & ".join(metric_keys) + " \\\\\n"
        f.write(f"        {header}")
        f.write("        \\midrule\n")
        
        # 写入每一个模型的数据
        for res in parsed_results:
            row_data = [res["method"]]
            for key in metric_keys:
                # 格式化保留 4 位小数
                val = res["metrics"].get(key, None)
                row_data.append(f"{val:.4f}" if val is not None else "-")
            row_str = " & ".join(row_data) + " \\\\\n"
            f.write(f"        {row_str}")
            
        # 写入表尾
        f.write("        \\bottomrule\n")
        f.write("    \\end{tabular}\n")
        f.write("\\end{table}\n")
        
    txt_logger.info(f"📊 LaTeX 表格已保存至: {tex_output_path}\n")


# ================= 主程序入口 =================
if __name__ == "__main__":
    mode  = "full"
    
    # 1. 定义 GT 路径
    GT_DIR_PATH = f'data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/test/{mode}/Landsat_02'
    
    METHODS_DICT = {
        "no_CA": f"/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/AdaMDR/inference_no_CA/{mode}/imgs/CIA/0/save_img",
        "no_SA": f"/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/AdaMDR/inference_no_SA/{mode}/imgs/CIA/0/save_img",
        "no_CA_SA": f"/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/AdaMDR/inference_no_CA_SA/{mode}/imgs/CIA/0/save_img",
        "no_MKIRA": f"/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/AdaMDR/inference_no_MKIRA/{mode}/imgs/CIA/0/save_img",
        "no_FiLM_inject": f"/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/AdaMDR/inference_no_FiLM_inject/{mode}/imgs/CIA/0/save_img",
        "position_1": f"/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/AdaMDR/inference_position_1/{mode}/imgs/CIA/0/save_img",
        "position_2": f"/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/AdaMDR/inference_position_2/{mode}/imgs/CIA/0/save_img",
        "position_123": f"/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/AdaMDR/inference_position_123/{mode}/imgs/CIA/0/save_img",
        "fusion_add": f"/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/fusion_meathod/inference_add/{mode}/imgs/CIA/0/save_img",
        "fusion_cat": f"/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/fusion_meathod/inference_cat/{mode}/imgs/CIA/0/save_img",
        "fusion_sub": f"/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/fusion_meathod/inference_sub/{mode}/imgs/CIA/0/save_img",
        "fusion_film": f"/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/fusion_meathod/inference_film/{mode}/imgs/CIA/0/save_img",
    }
    
    # 3. 定义输出目录和名称
    OUTPUT_DIRECTORY = "/home/zhaojiazhen/workspace/STF/STF/PPT/abl_ADAMDR/final_evaluation_summaries"
    OUTPUT_LOG_NAME = f"CIA_{mode}_evaluation_metrics_2.txt"
    
    # 4. 实例化指标列表 (保持你的参数设置)
    METRIC_LIST = [
        RMSE(is_reduce_channel=False),
        MAE(is_reduce_channel=False),
        PSNRONE(max_value=1.0, is_reduce_channel=False),
        SSIM(data_range=1.0, is_reduce_channel=False),
        ERGAS(ratio=1.0 / 16.0),
        CC(is_reduce_channel=False),
        SAM(),
        UIQI(is_reduce_channel=False),
    ]
    
    # 5. 执行批量评估
    batch_offline_evaluation_and_report(
        methods_dict=METHODS_DICT,
        gt_dir_path=GT_DIR_PATH,
        output_dir=OUTPUT_DIRECTORY,
        output_log_name=OUTPUT_LOG_NAME,
        metric_list=METRIC_LIST
    )