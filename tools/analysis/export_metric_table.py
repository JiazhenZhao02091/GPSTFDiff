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
root_dir = ''
table_patch = make_table(log_files_patch, root_dir)
table_full = make_table(log_files_full, root_dir)

print('Patch:')
print_table(table_patch)
print('\nFull:')
print_table(table_full)