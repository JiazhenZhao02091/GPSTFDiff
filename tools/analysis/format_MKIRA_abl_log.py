"""
功能:
    提取 MKIRA、AdaMDR 等消融实验日志末尾的指标，并生成 TXT 汇总和 LaTeX 表格。

参数:
    methods_dict (dict[str, str]): 方法名到 log.log 路径的映射。
    output_dir (str): 汇总结果保存目录。
    output_log_name (str): 输出 TXT 文件名。
    mode (str): 直接运行时脚本内配置的结果模式，例如 full 或 patch。

返回:
    None。函数直接写文件，不返回数据对象。

输出:
    输出同名 .txt 和 .tex 文件；日志缺失或无指标时在终端打印跳过信息。
"""
import os

def extract_metrics_and_generate_latex(methods_dict, output_dir, output_log_name):
    """
    遍历日志，提取最后一行的特定指标，并生成汇总 TXT 和 LaTeX 表格。
    
    参数:
        methods_dict (dict): { '方法名': '日志文件路径' }
        output_dir (str): 结果保存目录
        output_log_name (str): 汇总的 txt 结果日志名称 (例如 'compiled_metrics.txt')
    """
    # 1. 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建输出文件路径
    txt_output_path = os.path.join(output_dir, output_log_name)
    
    # 自动生成同名的 .tex 文件路径
    base_name = os.path.splitext(output_log_name)[0]
    tex_output_path = os.path.join(output_dir, f"{base_name}.tex")
    
    parsed_results = []
    metric_keys = []
    
    print(f"{'='*50}")
    print(f"🚀 开始提取结果日志...")
    print(f"{'='*50}")
    
    # 2. 遍历字典，提取数据
    for method_name, log_path in methods_dict.items():
        if not os.path.exists(log_path):
            print(f"⚠️ [跳过] 找不到日志文件: {log_path}")
            continue
            
        try:
            # 高效读取最后一行有效数据
            last_line = ""
            with open(log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        last_line = line.strip()
                        
            # 找到 RMSE 起始位置
            if "RMSE:" not in last_line:
                print(f"⚠️ [跳过] {method_name} 的日志最后一行未找到 RMSE 指标。")
                continue
                
            start_idx = last_line.find("RMSE:")
            metrics_str = last_line[start_idx:]  # 提取从 RMSE 开始的后半段字符串
            
            # --- 构造 TXT 输出所需格式 ---
            # 效果: "method_name, RMSE: 0.0268, MAE: 0.0160, ..."
            txt_line = f"{method_name}, {metrics_str}"
            
            # --- 构造 LaTeX 所需的数据结构 ---
            metrics_dict = {}
            items = metrics_str.split(", ")
            for item in items:
                if ": " in item:
                    k, v = item.split(": ")
                    metrics_dict[k.strip()] = v.strip()
                    
            # 记录表格列名（以第一个成功解析的模型拥有的指标为准）
            if not metric_keys:
                metric_keys = list(metrics_dict.keys())
                
            parsed_results.append({
                "method": method_name,
                "txt_line": txt_line,
                "metrics": metrics_dict
            })
            print(f"✅ 成功提取: {method_name}")
            
        except Exception as e:
            print(f"❌ 处理 {method_name} (路径: {log_path}) 时发生错误: {e}")

    # 3. 写入 TXT 结果汇总
    if parsed_results:
        with open(txt_output_path, 'w', encoding='utf-8') as f:
            for res in parsed_results:
                f.write(res["txt_line"] + "\n")
        print(f"\n📄 文本结果已保存至: {txt_output_path}")
    else:
        print("\n❌ 没有成功提取到任何数据，未生成文件。")
        return
    
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
                # 使用 .get 容错，如果某个模型缺少某个指标，填入 '-'
                row_data.append(res["metrics"].get(key, "-"))
            row_str = " & ".join(row_data) + " \\\\\n"
            f.write(f"        {row_str}")
            
        # 写入表尾
        f.write("        \\bottomrule\n")
        f.write("    \\end{tabular}\n")
        f.write("\\end{table}\n")
        
    print(f"📊 LaTeX 表格已保存至: {tex_output_path}\n")

# ================= 使用示例 =================
if __name__ == "__main__":
    # 模拟你的输入字典
    mode  = "full"
    my_methods_dict = {
        "no_CA": f"/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/AdaMDR/inference_no_CA/{mode}/txt_logs/log.log",
        "no_SA": f"/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/AdaMDR/inference_no_SA/{mode}/txt_logs/log.log",
        "no_CA_SA": f"/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/AdaMDR/inference_no_CA_SA/{mode}/txt_logs/log.log",
        "no_MKIRA": f"/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/AdaMDR/inference_no_MKIRA/{mode}/txt_logs/log.log",
        "no_FiLM_inject": f"/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/AdaMDR/inference_no_FiLM_inject/{mode}/txt_logs/log.log",
        "position_1": f"/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/AdaMDR/inference_position_1/{mode}/txt_logs/log.log",
        "position_2": f"/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/AdaMDR/inference_position_2/{mode}/txt_logs/log.log",
        "position_123": f"/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/AdaMDR/inference_position_123/{mode}/txt_logs/log.log",
        "fusion_add": f"/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/fusion_meathod/inference_add/{mode}/txt_logs/log.log",
        "fusion_cat": f"/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/fusion_meathod/inference_cat/{mode}/txt_logs/log.log",
        "fusion_sub": f"/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/fusion_meathod/inference_sub/{mode}/txt_logs/log.log",
        "fusion_film": f"/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/fusion_meathod/inference_film/{mode}/txt_logs/log.log",
    }
    
    OUTPUT_DIRECTORY = "/home/zhaojiazhen/workspace/STF/STF/artifacts/metric_log/abl/abl_ADAMDR"
    OUTPUT_LOG_FILENAME = f"{mode}_metrics_summary.txt"
    
    extract_metrics_and_generate_latex(
        methods_dict=my_methods_dict,
        output_dir=OUTPUT_DIRECTORY,
        output_log_name=OUTPUT_LOG_FILENAME
    )
