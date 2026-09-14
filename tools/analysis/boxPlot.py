"""
功能:
    计算 CIA patch 上多个预测模型的 RMSE、CC、SAM 分布，并绘制并排箱线图。

参数:
    true_dir (str): main 中配置的真实 patch 目录。
    pred_dirs_dict (dict[str, str]): 模型名称到预测 patch 目录的映射。
    out_dir (str): 箱线图输出目录。

返回:
    preprocess_to_tensor 返回形状为 (1, C, H, W) 的 Tensor。
    main 无返回值。

输出:
    在输出目录保存 BoxPlot_Metrics.png。
"""
import os
import glob
import numpy as np
import tifffile as tiff
import torch
import matplotlib.pyplot as plt
import sys
from matplotlib.ticker import MaxNLocator  # <--- 新增导入这个用于控制刻度密度的模块

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
    
    # 尺度自适应：超过 100 判定为万级别反射率 (CIA/LGC)，否则视为保持不变 (ML, 0-1)
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
    out_dir = 'artifacts/image/box_plot'
    
    # 你的预测结果字典
    pred_dirs_dict = {
        'GANSTFM': 'results/ganstfm/syy_setting-9/CIA/inference~Adam_1e-4/patch/imgs/CIA/save_img',
        'OPGAN': 'results/opgan/syy_setting-9/CIA/inference~RMSProp/patch/imgs/CIA/save_img',
        'SwinSTFM': 'results/swinstf/syy_setting-9/CIA/inference~Adam_1e-4~StepLR_step_size-15_gamma_5e-1/patch/imgs/CIA/save_img',
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
            
            # 精确寻找对应的预测文件
            pred_file = os.path.join(method_dir, f"{pred_base_name}.tif")
            if not os.path.exists(pred_file):
                # 兼容 .tiff 后缀
                pred_file = os.path.join(method_dir, f"{pred_base_name}.tiff")
                if not os.path.exists(pred_file):
                    # 如果还是找不到，说明这个 Patch 没生成，直接跳过
                    continue
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
    # 1. 调整画布比例，使其更扁宽 (例如每个子图宽5.5，高4.5)
    fig, axes = plt.subplots(nrows=1, ncols=num_metrics, figsize=(5.5 * num_metrics, 4.5))
    if num_metrics == 1:
        axes = [axes]
    
    labels_abc = ['(a)', '(b)', '(c)', '(d)', '(e)']
    
    for i, (m_name, _) in enumerate(metrics.items()):
        ax = axes[i]
        
        data_to_plot = []
        for method in methods:
            d = results[m_name][method]
            
            # ================= 修改 3：绘图前清洗 NaN 数据 =================
            d_clean = [val for val in d if not np.isnan(val)]
            
            if len(d_clean) == 0:
                print(f"[警告] {method} 的 {m_name} 没有计算出任何有效数据！")
                data_to_plot.append([0])
            else:
                data_to_plot.append(d_clean)
            # ===============================================================
        
        # 2. 精调箱体样式：变窄(widths=0.45)、隐藏离群点(showfliers=False)、加粗线条
        box = ax.boxplot(data_to_plot, tick_labels=methods, patch_artist=True, 
                         widths=0.45, showfliers=False,
                         boxprops=dict(facecolor='white', color='black', linewidth=1.2),
                         medianprops=dict(color='darkorange', linewidth=2.0),
                         whiskerprops=dict(color='black', linewidth=1.2),
                         capprops=dict(color='black', linewidth=1.2))
        
        ax.set_ylabel(m_name, fontsize=16)
        # 3. 补回 X 轴标签 "Method"
        ax.set_xlabel('Method', fontsize=16)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        
        # ================= 新增：让 Y 轴标尺刻度更密集 =================
        # nbins 限定了刻度划分的最大区间数，数字越大刻度越密集（默认通常是4~5）
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        # ===============================================================
        
        # 4. 调整 (a)(b)(c) 的垂直距离 (y=-0.35)，防止和 'Method' 标签重叠
        ax.text(0.5, -0.35, labels_abc[i], transform=ax.transAxes, 
                fontsize=18, ha='center', va='top')

    # 5. 增加子图间距以及底部留白以容纳文本
    plt.subplots_adjust(wspace=0.3, bottom=0.28)
    
    save_path = os.path.join(out_dir, 'BoxPlot_Metrics.png')
    plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"箱线图已保存至: {save_path}")

if __name__ == '__main__':
    main()
