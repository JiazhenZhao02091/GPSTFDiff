"""
功能:
    从 lap、DDPM、DDIM 的离线指标日志中提取最终 Average 指标，并打印 Markdown 表格。

参数:
    log_path (str): extract_metrics 读取的单个日志路径。
    log_files_patch (list[tuple[str, str]]): patch 日志文件及模型名配置。
    log_files_full (list[tuple[str, str]]): full 日志文件及模型名配置。
    root_dir (str): 日志根目录。

返回:
    extract_metrics 返回 8 个浮点指标；未找到指标时返回 None。
    make_table 返回 Markdown 表格使用的二维列表。

输出:
    直接运行时在终端输出 Patch 和 Full 两张 Markdown 指标表。
"""
import re

def extract_metrics(log_path):
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 找到最后一行平均值
    avg_line = ''
    for line in reversed(lines):
        if 'Average over' in line:
            avg_line = line
            break
    if not avg_line:
        return None
    # 提取数值
    pattern = (r'RMSE: ([\d\.]+).*?MAE: ([\d\.]+).*?PSNRONE: ([\d\.]+).*?SSIM: ([\d\.]+)'
               r'.*?ergas: ([\d\.]+).*?CC: ([\d\.]+).*?SAM: ([\d\.]+).*?UIQI: ([\d\.]+)')
    match = re.search(pattern, avg_line)
    if not match:
        return None
    return [float(x) for x in match.groups()]

# 文件名与模型名对应关系
log_files_patch = [
    ('lap', 'offline_metric_cal_lap_patch.log'),
    ('DDPM', 'offline_metric_cal_DDPM_patch.log'),
    ('DDIM', 'offline_metric_cal_DDIM_patch.log'),
]
log_files_full = [
    ('lap', 'offline_metric_cal_lap_full.log'),
    ('DDPM', 'offline_metric_cal_DDPM_full.log'),
    ('DDIM', 'offline_metric_cal_DDIM_full.log'),
]

def make_table(log_files, root_dir):
    table = []
    for model, fname in log_files:
        metrics = extract_metrics(f'{root_dir}/{fname}')
        if metrics:
            table.append([model] + [f'{v:.4f}' for v in metrics] + [''])
    return table

def print_table(table):
    header = ['Sampler', 'RMSE', 'MAE', 'PSNRONE', 'SSIM', 'ergas', 'CC', 'SAM', 'UIQI', 'times']
    print('| ' + ' | '.join(header) + ' |')
    print('|' + '--------|' * len(header))
    for row in table:
        print('| ' + ' | '.join(row) + ' |')

# 用法示例
root_dir = '/home/zhaojiazhen/workspace/STF/STF/artifacts/metric_log/abl'
table_patch = make_table(log_files_patch, root_dir)
table_full = make_table(log_files_full, root_dir)

print('Patch:')
print_table(table_patch)
print('\nFull:')
print_table(table_full)
