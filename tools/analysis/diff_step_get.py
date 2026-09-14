"""
功能:
    汇总 diff_step 消融实验日志中的计时信息和最终验证指标。

参数:
    log_path (str): extract_log_info 读取的单个 log.log 路径。
    root_dir (str): 直接运行时扫描的 diff_step 实验根目录。
    output_file (str): 汇总日志输出路径。

返回:
    extract_log_info 返回 (timing, summary) 字符串元组。

输出:
    直接运行时写出 summary_log.txt，并在终端打印保存路径。
"""
import os

def extract_log_info(log_path):
    timing = ""
    summary = ""
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if "计时开始" in line or "计时结束" in line:
                timing += line.strip() + "\n"
        # 最后一行汇总指标
        for line in reversed(lines):
            if "val epoch: 0, loss:" in line:
                summary = line.strip()
                break
    return timing, summary

root_dir = "/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/diff_step"
output_file = "/home/zhaojiazhen/workspace/STF/STF/artifacts/metric_log/summary_log.txt"

with open(output_file, 'w', encoding='utf-8') as out_f:
    for method in os.listdir(root_dir):
        method_path = os.path.join(root_dir, method)
        if not os.path.isdir(method_path):
            continue
        for step in os.listdir(method_path):
            step_path = os.path.join(method_path, step)
            if not os.path.isdir(step_path):
                continue
            for mode in ['full', 'patch']:
                log_path = os.path.join(step_path, mode, 'txt_logs', 'log.log')
                if os.path.exists(log_path):
                    timing, summary = extract_log_info(log_path)
                    # 组织名称格式
                    name = f"{method}_{step}_{mode}"
                    out_f.write(f"{name}:\n")
                    out_f.write(timing)
                    out_f.write(f"{name}_summary: {summary}\n\n")

print(f"日志已保存到 {output_file}")
