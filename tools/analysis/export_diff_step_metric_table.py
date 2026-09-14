"""
功能:
    批量读取不同采样步数实验的离线指标日志，并导出 full/patch 两套 CSV 指标表。

参数:
    log_path (str): extract_metrics 读取的单个日志路径。
    folder (str): collect_logs 扫描的日志目录。
    folders (dict[str, list[str]]): 直接运行时使用的 full/patch 日志子目录配置。

返回:
    extract_metrics 返回 8 个指标组成的列表；解析失败时返回 [None] * 8。
    collect_logs 返回每个日志文件的 step 和指标列表。

输出:
    直接运行时生成 full_metrics.csv 和 patch_metrics.csv。
"""
import os
import re
import pandas as pd

BASE_DIR = '/home/zhaojiazhen/workspace/STF/STF/artifacts/metric_log/abl'

def extract_metrics(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()
    for line in reversed(lines):
        if 'Average over' in line:
            # 提取指标
            pattern = (r'RMSE: ([\d\.]+).*MAE: ([\d\.]+).*PSNRONE: ([\d\.]+).*SSIM: ([\d\.]+).*'
                       r'ergas: ([\d\.]+), CC: ([\d\.]+).*SAM: ([\d\.]+), UIQI: ([\d\.]+)')
            match = re.search(pattern, line)
            if match:
                return [float(x) for x in match.groups()]
    return [None]*8

def collect_logs(folder):
    data = []
    for fname in os.listdir(folder):
        if fname.endswith('.log'):
            step = os.path.splitext(fname)[0]
            metrics = extract_metrics(os.path.join(folder, fname))
            data.append([step] + metrics)
    return data

folders = {
    'full': ['lap_full', 'DDIM_full'],
    'patch': ['lap_patch', 'DDIM_patch']
}

for key, subfolders in folders.items():
    all_data = []
    for subfolder in subfolders:
        folder_path = os.path.join(BASE_DIR, subfolder)
        all_data += collect_logs(folder_path)
    df = pd.DataFrame(all_data, columns=['step', 'RMSE', 'MAE', 'PSNRONE', 'SSIM', 'ergas', 'CC', 'SAM', 'UIQI'])
    df.to_csv(os.path.join(BASE_DIR, f'{key}_metrics.csv'), index=False)
