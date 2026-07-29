import os
import glob
import numpy as np
import tifffile as tiff
import torch
import matplotlib.pyplot as plt
import sys
from matplotlib.ticker import MaxNLocator  # <--- 新增导入这个用于控制刻度密度的模块
from scipy.stats import norm  # <--- 新增导入用于正态分布拟合

# 引入项目指标，如果运行报错 ModuleNotFoundError，请确保在 STF 根目录执行脚本
# 或通过 sys.path.append() 添加根目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.metrics import RMSE, CC, SAM

# ================= 新增：全局字体设置 =================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 14
# ====================================================

def preprocess_to_tensor(img_array):
    """
    处理图像维度并统一归一化到 [0, 1]
    输出 Tensor 维度: (1, C, H, W)
    """
    img = img_array.astype(np.float32)
    
    # 尺度自适应：超过 100 判定为万级别反射率 (ML/ML)，否则视为保持不变 (ML, 0-1)
    if np.max(img) > 100:
        img = img / 10000.0
        
    # 统一转换到 (C, H, W)
    if img.ndim == 3:
        if img.shape[-1] <= 10:  # 假设通道数较小，如果在最后一维则转置
            img = img.transpose(2, 0, 1)
    
    # 转为 tensor 并增加 batch 维
    tensor = torch.from_numpy(img).unsqueeze(0)
    return tensor

def get_file_prefix(filename):
    basename = os.path.basename(filename)
    parts = basename.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return basename.split('.')[0]

def main():
    # ================= 目录与参数配置 =================
    true_dir = 'data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/test/patch/Landsat_02'
    out_dir = 'PPT/box_plot/test/CIA_4'
    
    # 你的预测结果字典
    # pred_dirs_dict = {
    #     'GANSTFM': 'results/ganstfm/syy_setting-9/ML/inference~Adam_1e-4/patch/imgs/ML/save_img',
    #     'OPGAN': 'results/opgan/syy_setting-9/ML/inference~RMSProp/patch/imgs/ML/save_img',
    #     'SwinSTFM': 'results/swinstf/syy_setting-9/ML/inference~Adam_1e-4~StepLR_step_size-15_gamma_5e-1/patch/imgs/ML/save_img',
    # }
    
    pred_dirs_dict = {
                'STARFM': 'results/starfm/syy_setting~9/CIA/one_pair~patch_size_120~patch_stride_50~window_size_51~num_classes_20/patch/imgs/CIA/save_img',
                # 'FSDAF': 'results/FSDAF/CIA/patch',
                # 'FitFC': 'results/FitFC/CIA/patch/FitFC_Results_L2',
                # 'stfdcnn': 'results/stfdcnn/syy_setting~9/CIA/inference~stage_1_SGD_1e-2~stage_2_SGD_1e-1/patch/imgs/stage_2/CIA/save_img',
                # 'ganstfm': 'results/ganstfm/syy_setting-9/CIA/inference~Adam_1e-4/patch/imgs/CIA/save_img',
                'STFGAN': 'results/stfgan/syy_setting-9/CIA/inference~stage_1~RMSProp~stage_2~RMSProp/patch/imgs/stage_2/CIA/save_img',
                # 'opgan': 'results/opgan/syy_setting-9/CIA/inference~RMSProp/patch/imgs/CIA/save_img',
                'SwinSTFM': 'results/swinstf/syy_setting-9/CIA/inference~Adam_1e-4~StepLR_step_size-15_gamma_5e-1/patch/imgs/CIA/save_img',
                # 'fsdformer': 'results/fsdformer/syy_setting-9/CIA/inferencer/patch/imgs/CIA/save_img',
                'STFMamba': 'results/stfmamba/syy_setting-9/CIA/inferencer/patch/imgs/CIA/save_img',
                'STFDiff': 'results/LapSTFDiff/lap/syy_setting-9/CIA/inference/inferency_8/patch/imgs/CIA/0/save_img',
                'LapSTFDiff': 'results/stfdiff/syy_setting-9/model6_GN_SiLU/CIA/inference/patch/imgs/CIA/0/save_img',
                
    }
    os.makedirs(out_dir, exist_ok=True)
    
    # 初始化指标
    metrics = {
        'RMSE': RMSE(is_reduce_channel=False),
        'CC': CC(is_reduce_channel=False),
        'SAM': SAM()
    }
    
    methods = list(pred_dirs_dict.keys())
    # results[metric_name][method_name] = [val1, val2, ...]
    results = {m_name: {method: [] for method in methods} for m_name in metrics.keys()}

    valid_extensions = ('.tif', '.tiff')
    true_files = [f for f in glob.glob(os.path.join(true_dir, '*')) if f.lower().endswith(valid_extensions)]
    
    print(f"共发现 {len(true_files)} 张真实图像，开始计算指标...\n")

    # ================= 计算指标 =================
    for true_file in true_files:
        # 获取不带后缀的真实文件名，例如 'Group_01_L_0_256_0_256'
        base_name = os.path.splitext(os.path.basename(true_file))[0]
        
        try:
            true_img_np = tiff.imread(true_file)
            true_tensor = preprocess_to_tensor(true_img_np)
        except Exception as e:
            print(f"读取真实图像失败: {true_file}, {e}")
            continue

        # ================= 修改 1：绝对无效背景过滤 =================
        if torch.max(true_tensor) <= 1e-5:
            # 如果真实 Patch 全是黑边或无效填充，没有任何地物信息，直接跳过此 Patch
            continue
        
        # 提取真实图像的标准差，用于判断是否为绝对平滑区域
        std_true = torch.std(true_tensor).item()
        # ==========================================================

        for method in methods:
            method_dir = pred_dirs_dict[method]
            if not os.path.exists(method_dir): continue
            
            # ================= 核心修复：精准构建预测文件名 =================
            # 真实文件：Group_01_L_0_256_0_256.tif
            # 预测文件：Group_01_save_img_0_256_0_256.tif
            # 我们直接把 '_L_' 替换为 '_save_img_' 即可实现空间坐标的精准对齐
            pred_base_name = base_name.replace('_L_', '_save_img_')
            
            # 提取空间坐标部分
            coords = '_'.join(base_name.split('_')[3:])  # 取最后四段
            # 在预测目录下查找包含该坐标的文件
            pred_candidates = glob.glob(os.path.join(method_dir, f"*{coords}.tif")) + \
                  glob.glob(os.path.join(method_dir, f"*{coords}.tiff"))
            if len(pred_candidates) == 0:
                # 如果没找到就跳过
                continue
            pred_file = pred_candidates[0]  # 取第一个匹配到的文件
            # =================================================================
                
            try:
                pred_img_np = tiff.imread(pred_file)
                # 检查尺寸是否一致
                if true_img_np.shape != pred_img_np.shape:
                    continue
                pred_tensor = preprocess_to_tensor(pred_img_np)
            except Exception:
                continue

            # 使用 torch 计算指标
            for m_name, metric_func in metrics.items():
                # ================= 修改 2：CC 专属低方差保护 =================
                if m_name == 'CC' and std_true < 1e-3:
                    # 如果该区域极其平滑（如纯水体），CC 容易被噪声主导算崩，赋值为 NaN
                    results[m_name][method].append(np.nan)
                else:
                    val = metric_func(true_tensor, pred_tensor)
                    # 将多波段或张量结果转为单张图像的平均标量值
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    else:
                        val = np.mean(val)
                    results[m_name][method].append(val)
                # =============================================================
    # ================= 开始绘图 =================
    print("\n指标计算完成，开始绘制箱线图...")

    num_metrics = len(metrics)
    num_methods = len(methods)
    
    # 画布：将宽度大幅增加，高度适当压缩，实现约 2:1 的宽高比
    # 比如总高度设为重叠箱线图合适的 10，则宽度设为 20
    fig, axes = plt.subplots(nrows=num_metrics, ncols=1, figsize=(13, 10))
    if num_metrics == 1:
        axes = [axes]

    # ================= 修改区 1: 颜色配置 =================
    import matplotlib.cm as cm
    # 先获取一个基础对比度较柔和的色带 (如 set2 或 tab20)
    color_map = cm.get_cmap('Set3', num_methods)
    colors = [color_map(i) for i in range(num_methods)]
    
    # 针对性修改特定方法的颜色
    for j, method in enumerate(methods):
        if method == 'LapSTFDiff':
            colors[j] = '#E63946'  # 非常明亮突出的红色
        elif method == 'STFDiff':
            colors[j] = '#457B9D'  # 深蓝色进行对比
    # ======================================================

    for i, (m_name, _) in enumerate(metrics.items()):
        ax = axes[i]
        data_to_plot = []
        for method in methods:
            d = results[m_name][method]
            d_clean = [val for val in d if not np.isnan(val)]
            if len(d_clean) == 0:
                print(f"[警告] {method} 的 {m_name} 没有计算出任何有效数据！")
                data_to_plot.append([0])
            else:
                data_to_plot.append(d_clean)

        for j, (method, data) in enumerate(zip(methods, data_to_plot)):
            pos = j + 1
            color = colors[j]
            # 1. 箱线图 (稍微调窄箱子宽度 width: 0.25 -> 0.22 让画面更透气)
            bplot = ax.boxplot(data, positions=[pos], widths=0.15, 
                               showfliers=False, patch_artist=True,
                               boxprops=dict(facecolor=color, color=color, alpha=0.9),
                               medianprops=dict(color='white', linewidth=2),
                               whiskerprops=dict(color=color, linewidth=1.5),
                               capprops=dict(color=color, linewidth=1.5))
            # 2. 散点 (适当增加散点的水平分布抖动范围 0.04 -> 0.06，或者保持 0.04)
            x_scatter = np.random.normal(pos + 0.25, 0.04, size=len(data))
            ax.scatter(x_scatter, data, color=color, alpha=0.9, s=5, zorder=3)
            # 3. 正态分布曲线
            if len(data) > 1:
                mu, std = norm.fit(data)
                if not np.isnan(mu) and not np.isnan(std) and std > 0:
                    y_curve = np.linspace(min(data), max(data), 100)
                    p = norm.pdf(y_curve, mu, std)
                    x_curve = pos + 0.25 + (p / p.max()) * 0.15 
                    ax.plot(x_curve, y_curve, color=color, linewidth=1.5, zorder=4)

        ax.set_xticks(range(1, num_methods + 1))
        # 恢复每个图的 X 轴方法标签并稍微缩小字号防止重叠
        ax.set_xticklabels(methods, fontsize=14)
        ax.set_ylabel(m_name, fontsize=16)
        
        # 去掉原本可能有的 "Method" xlabel
        ax.set_xlabel('')
        
        ax.tick_params(axis='y', labelsize=15)
        ax.tick_params(axis='x', pad=5) # 标签稍微远离坐标轴一点点
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # ================= 调整区：间距控制 =================
    # hspace 调小 (拉近行距)，因为现在图变宽了，底部的字也不会挤到一块去
    plt.subplots_adjust(hspace=0.15, bottom=0.1)
    
    save_path = os.path.join(out_dir, 'BoxPlot_Metrics_Updated_CIA.png')
    plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='white')

    save_path = os.path.join(out_dir, 'BoxPlot_Metrics_Updated_CIA.pdf')
    plt.savefig(save_path, facecolor='white', bbox_inches='tight')

    plt.close()
    print(f"箱线图已保存至: {save_path}")

if __name__ == '__main__':
    main()