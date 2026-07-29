import os
import sys
import re
import glob
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches

# 尝试导入 syy_setting 以获取年份信息
try:
    from scripts.spatio_temparol_fusion.dataset_generation.dataset_config.syy_setting import DATASET_SETTING
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    try:
        from scripts.spatio_temparol_fusion.dataset_generation.dataset_config.syy_setting import DATASET_SETTING
    except ImportError:
        print("Warning: Could not import DATASET_SETTING from syy_setting.py")
        DATASET_SETTING = {}

# ================= 全局绘图设置 =================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 15

# ================= 颜色与分组策略 =================
def get_color_for_method(method_name):
    """
    根据模型族灵活分配颜色，低饱和背景 + 高亮突出创新方案
    """
    m = method_name.lower()
    # 传统方法 (低饱和灰色)
    if m in ['starfm', 'fsdaf', 'fitfc']:
        return '#B0B0B0'
    # CNN/早期GAN系列 (低饱和蓝色)
    elif m in ['stfdcnn', 'opgan', 'stfgan']:
        return '#6BAED6'
    # Transformer/Mamba/Diffusion 等新架构 (低饱和绿色)
    elif m in ['fsdformer', 'stfmamba', 'stfdiff']:
        return '#74C476'
    # ★ 强化的提出/关注方法 (高亮橘红色) ★
    elif m in ['ganstfm', 'swinstf', 'lapstfdiff', 'proposed']:
        return '#F56C42'  # 橘红色
    # 默认颜色
    return '#999999'

# ================= 工具函数 =================
def preprocess_image(img_array):
    img = img_array.astype(np.float32)
    if np.max(img) > 100: img = img / 10000.0
    if img.ndim == 2: img = img[np.newaxis, ...]
    elif img.ndim == 3:
        if img.shape[-1] <= 16: img = img.transpose(2, 0, 1)
    else: raise ValueError(f"Unsupported ndim: {img.ndim}")
    return img

def preprocess_mask(mask_array):
    mask = np.asarray(mask_array)
    if mask.ndim == 3:
        if mask.shape[0] == 1: mask = mask[0]
        elif mask.shape[-1] == 1: mask = mask[..., 0]
        else: raise ValueError(f"CDL mask should be single-channel, got shape={mask.shape}")
    return mask.astype(np.int32)

def safe_corrcoef(x, y, eps=1e-8):
    if x.size < 2 or y.size < 2: return np.nan
    x_std, y_std = np.std(x), np.std(y)
    if x_std < eps or y_std < eps: return np.nan
    return np.corrcoef(x, y)[0, 1]

def compute_masked_metrics(true_img, pred_img, crop_mask, eps=1e-8):
    assert true_img.shape == pred_img.shape
    assert true_img.shape[1:] == crop_mask.shape
    valid_idx = crop_mask
    n_valid = int(np.sum(valid_idx))
    if n_valid == 0: return None

    true_pixels = true_img[:, valid_idx]
    pred_pixels = pred_img[:, valid_idx]

    bias = np.mean(pred_pixels - true_pixels)
    rmse = np.sqrt(np.mean((pred_pixels - true_pixels) ** 2))

    cc_list = []
    for c in range(true_pixels.shape[0]):
        cc_c = safe_corrcoef(true_pixels[c], pred_pixels[c], eps=eps)
        if not np.isnan(cc_c): cc_list.append(cc_c)
    cc = np.mean(cc_list) if len(cc_list) > 0 else np.nan

    dot = np.sum(true_pixels * pred_pixels, axis=0)
    norm_true = np.linalg.norm(true_pixels, axis=0)
    norm_pred = np.linalg.norm(pred_pixels, axis=0)
    denom = np.maximum(norm_true * norm_pred, eps)

    cos_theta = np.clip(dot / denom, -1.0, 1.0)
    sam = np.degrees(np.mean(np.arccos(cos_theta)))
    return {'Bias': bias, 'RMSE': rmse, 'CC': cc, 'SAM': sam}

def find_file_by_base(directory, base_name):
    for ext in ['.tif', '.tiff']:
        f = os.path.join(directory, base_name + ext)
        if os.path.exists(f): return f
    return None

def build_pred_base_name(true_base_name, replace_rule=('_L_', '_save_img_')):
    src_token, dst_token = replace_rule
    return true_base_name.replace(src_token, dst_token)

def build_cdl_base_name(true_base_name, year):
    if year is None: return true_base_name
    parts = true_base_name.split('_')
    if len(parts) >= 7 and "Group" in parts[0]:
        coords_suffix = '_'.join(parts[3:])
        return f"CDL{year}_{coords_suffix}"
    return f"CDL{year}_{true_base_name}"

def get_year_from_filename(filename):
    match = re.search(r'(\d{4})', filename)
    return match.group(1) if match else None

def get_inference_config(location, split='val', setting_dict=DATASET_SETTING):
    if location not in setting_dict: return []
    triplets = setting_dict[location].get(split, [])
    dataset_split = []
    for idx, triplet in enumerate(triplets):
        target_date = triplet[1] if len(triplet) >= 2 else (triplet[0] if len(triplet) == 1 else None)
        if not target_date: continue
        group_id = f"Group_{idx+1:02d}"
        dataset_split.append({'group_id': group_id, 'year': get_year_from_filename(target_date) or target_date.split('-')[0]})
    return dataset_split

# ================= 主流程 =================
def main():
    LOCATION, SPLIT = 'ML', 'test'
    
    true_dir = 'data/spatio_temporal_fusion/ML/private_data/syy_setting-9/test/patch/Landsat_02'
    cdl_dir = 'data/spatio_temporal_fusion/ML/CDL_patch'
    out_dir = 'PPT/box_plot_cropwise'
    os.makedirs(out_dir, exist_ok=True)

    # 预留了 12 个方法位（请根据你的真实路径自行调整确认）
    pred_dirs_dict = {
        'starfm': 'results/starfm/syy_setting~9/ML/one_pair~patch_size_120~patch_stride_50~window_size_51~num_classes_20/patch/imgs/ML/save_img',
        'FSDAF': 'results/FSDAF/ML/patch',
        'FitFC': 'results/FitFC/ML/patch/FitFC_Results_L2',
        'stfdcnn': 'results/stfdcnn/syy_setting~9/ML/inference~stage_1_SGD_1e-2~stage_2_SGD_1e-1/patch/imgs/stage_2/ML/save_img',
        'ganstfm': 'results/ganstfm/syy_setting-9/ML/inference~Adam_1e-4/patch/imgs/ML/save_img',
        'stfgan': 'results/stfgan/syy_setting-9/ML/inference~stage_1~RMSProp~stage_2~RMSProp/patch/imgs/stage_2/ML/save_img',
        'opgan': 'results/opgan/syy_setting-9/ML/inference~RMSProp/patch/imgs/ML/save_img',
        'swinstf': 'results/swinstf/syy_setting-9/ML/inference~Adam_1e-4~StepLR_step_size-15_gamma_5e-1/patch/imgs/ML/save_img',
        'fsdformer': 'results/fsdformer/syy_setting-9/ML/inferencer/patch/imgs/ML/save_img',
        'stfmamba': 'results/stfmamba/syy_setting-9/ML/inferencer/patch/imgs/ML/save_img',
        'stfdiff': 'results/stfdiff/syy_setting-9/model6_GN_SiLU/ML/inference/patch/imgs/ML/0/save_img',
        'LapSTFDiff': 'results/LapSTFDiff/lap/syy_setting-9/ML/inference/patch/imgs/ML/0/save_img',
    }
    pred_replace_rule = ('_L_', '_save_img_')

    dataset_split = get_inference_config(LOCATION, split=SPLIT)
    group_to_year = {item['group_id']: item['year'] for item in dataset_split}
    if not group_to_year:
        group_to_year = {'Group_01': '2021', 'Group_02': '2022', 'Group_03': '2023', 'Group_04': '2023'}

    crop_value_map = {'Corn': [1], 'Soybean': [5], 'Grassland': [176], 'Forest': [141, 143], 'Developed': [121, 122, 123, 124]}
    metrics_to_plot = ['Bias', 'RMSE', 'CC', 'SAM']
    methods, crops = list(pred_dirs_dict.keys()), list(crop_value_map.keys())

    results = {crop: {metric: {method: [] for method in methods} for metric in metrics_to_plot} for crop in crops}
    valid_extensions = ('.tif', '.tiff')
    true_files = [f for f in glob.glob(os.path.join(true_dir, '*')) if f.lower().endswith(valid_extensions)]

    print(f'共发现 {len(true_files)} 张真实 patch，开始进行 crop-wise 统计...\n')

    for true_file in true_files:
        true_base_name = os.path.splitext(os.path.basename(true_file))[0]
        match_group = re.search(r'(Group_\d+)', true_base_name)
        group_id = match_group.group(1) if match_group else None
        year = group_to_year.get(group_id)

        cdl_base_name = build_cdl_base_name(true_base_name, year)
        cdl_file = find_file_by_base(cdl_dir, cdl_base_name)
        if cdl_file is None: continue

        try:
            true_img = preprocess_image(tiff.imread(true_file))
            cdl_mask = preprocess_mask(tiff.imread(cdl_file))
        except Exception: 
            continue

        if true_img.shape[1:] != cdl_mask.shape or np.max(true_img) <= 1e-5: continue

        missing_any_method = False
        valid_pred_imgs = {}
        for method in methods:
            pred_file = find_file_by_base(pred_dirs_dict[method], build_pred_base_name(true_base_name, pred_replace_rule))
            # 兼容：如果严格名字匹配失败，允许存在即可 (此处简化为确保预测试图可运行)
            if pred_file is None: continue
            try:
                pred_img = preprocess_image(tiff.imread(pred_file))
                valid_pred_imgs[method] = pred_img
            except Exception:
                missing_any_method = True; break
        
        # 为了展示逻辑，即使少量方法没找到也尽量容忍记录
        if len(valid_pred_imgs) == 0: continue

        for crop in crops:
            crop_mask = np.isin(cdl_mask, crop_value_map[crop])
            if int(np.sum(crop_mask)) < max(30, 0.05 * cdl_mask.shape[0] * cdl_mask.shape[1]): continue
            
            for method, pred_img in valid_pred_imgs.items():
                if pred_img.shape != true_img.shape: continue
                metric_dict = compute_masked_metrics(true_img, pred_img, crop_mask)
                if metric_dict and not np.isnan(metric_dict['CC']):
                    for m_name in metrics_to_plot:
                        results[crop][m_name][method].append(metric_dict[m_name])

    print('\n统计完成，开始绘制分块转置箱线图...')

    # ================= 绘图核心重组区 =================
    # 根据指标属性将 4 个指标分为两份表进行排版
    layout_graphs = {
        'Figure_1_Error_Metrics': ['Bias', 'RMSE'],   # 误差图 2 × 5
        'Figure_2_Quality_Metrics': ['CC', 'SAM']     # 光谱与质量图 2 × 5
    }

    for fig_id, (fig_name, split_metrics) in enumerate(layout_graphs.items()):
        nrows = len(split_metrics)
        ncols = len(crops)
        
        # 1. 扩宽画布至极端水平（适配 12 个长标签），设置每一行的Y轴共享以便横向对比同类指标
        fig, axes = plt.subplots(nrows, ncols, figsize=(26, 12), sharey='row', squeeze=False)
        
        for r, metric_name in enumerate(split_metrics):
            for c, crop in enumerate(crops):
                ax = axes[r, c]
                
                # 收集当前格子的箱线数据
                data_to_plot = []
                for method in methods:
                    vals = results[crop][metric_name].get(method, [])
                    vals = [v for v in vals if not np.isnan(v)]
                    data_to_plot.append(vals if len(vals) > 0 else [0.0])
                
                # 2. 个性化箱线颜色与绘制
                bplot = ax.boxplot(
                    data_to_plot,
                    widths=0.6,
                    showfliers=False,      # 这里隐藏离群点使图形更清爽，可按需开放
                    patch_artist=True,
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(color='black', linewidth=1.0),
                    capprops=dict(color='black', linewidth=1.0)
                )

                # 将指定的分组颜色赋予对应的箱子盒
                for patch, method in zip(bplot['boxes'], methods):
                    clr = get_color_for_method(method)
                    patch.set_facecolor(clr)
                    patch.set_alpha(0.8)

                # 3. 标签与抬头处理
                if r == 0:
                    # 将种类名称(Crop)放在第一排题头上
                    ax.set_title(crop, fontsize=22, fontweight='bold', pad=15)
                
                if c == 0:
                    # 将指标名称(Metric)放在第一列左侧
                    ax.set_ylabel(metric_name, fontsize=20, fontweight='bold')
                
                # 4. X 轴精简逻辑 (隐藏冗余轴)
                ax.set_xticks(range(1, len(methods) + 1))
                if r == nrows - 1:
                    # 最底下一行：显示倾斜标签
                    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=16)
                else:
                    # 上面的子图：隐去 X 轴文字，以节省视觉拥挤度
                    ax.set_xticklabels([])
                    ax.tick_params(axis='x', length=0)
                
                ax.tick_params(axis='y', labelsize=16)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
                
                # 可选增加网格线对齐
                ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # 画布间距收缩
        plt.subplots_adjust(wspace=0.05, hspace=0.15, bottom=0.15)
        
        # 全局图例：在整个 Figure 底部或顶部横向生成颜色指引 (可选，让图片专业度飙升)
        legend_elements = [
            mpatches.Patch(color='#B0B0B0', label='Traditional Rules'),
            mpatches.Patch(color='#6BAED6', label='CNN / D-GANs'),
            mpatches.Patch(color='#74C476', label='Trans-Models / Diffs'),
            mpatches.Patch(color='#F56C42', label='Proposed (SwinSTFM/LapSTFDiff)')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=18, frameon=False)

        save_path = os.path.join(out_dir, f'{fig_name}.png')
        plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ 排版方案已保存至: {save_path}")

if __name__ == '__main__':
    main()