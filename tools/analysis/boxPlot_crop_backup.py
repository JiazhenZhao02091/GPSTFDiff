import os
import sys
import re
import glob
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors

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
plt.rcParams['font.size'] = 14  # 稍微调大全局基础字体

# ================= 工具函数 =================
def preprocess_image(img_array):
    """
    将影像统一到 float32 + (C, H, W) + [0, 1]
    """
    img = img_array.astype(np.float32)

    # 若是万级反射率，归一化到 [0, 1]
    if np.max(img) > 100:
        img = img / 10000.0

    # 转成 (C, H, W)
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    elif img.ndim == 3:
        # 若通道在最后一维
        if img.shape[-1] <= 16:
            img = img.transpose(2, 0, 1)
    else:
        raise ValueError(f"Unsupported image ndim: {img.ndim}")

    return img


def preprocess_mask(mask_array):
    """
    将 CDL 掩膜统一到 (H, W)，保持整数类别编码
    """
    mask = np.asarray(mask_array)

    if mask.ndim == 3:
        # 常见情况：单通道但带一个冗余维
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            raise ValueError(f"CDL mask should be single-channel, got shape={mask.shape}")

    return mask.astype(np.int32)


def safe_corrcoef(x, y, eps=1e-8):
    """
    一维向量 Pearson 相关系数，低方差时返回 np.nan
    """
    if x.size < 2 or y.size < 2:
        return np.nan

    x_std = np.std(x)
    y_std = np.std(y)
    if x_std < eps or y_std < eps:
        return np.nan

    return np.corrcoef(x, y)[0, 1]


def compute_masked_metrics(true_img, pred_img, crop_mask, eps=1e-8):
    """
    在 crop_mask=True 的像元上计算指标
    true_img, pred_img: (C, H, W), float32, [0, 1]
    crop_mask: (H, W), bool

    返回:
        {
            'Bias': float,
            'RMSE': float,
            'CC': float,
            'SAM': float,   # degree
        }
    """
    assert true_img.shape == pred_img.shape
    assert true_img.shape[1:] == crop_mask.shape

    valid_idx = crop_mask
    n_valid = int(np.sum(valid_idx))
    if n_valid == 0:
        return None

    # 展平到 (C, N)
    true_pixels = true_img[:, valid_idx]
    pred_pixels = pred_img[:, valid_idx]

    # ---- Bias ----
    bias = np.mean(pred_pixels - true_pixels)

    # ---- RMSE ----
    rmse = np.sqrt(np.mean((pred_pixels - true_pixels) ** 2))

    # ---- CC（按通道分别算，再取平均）----
    cc_list = []
    for c in range(true_pixels.shape[0]):
        cc_c = safe_corrcoef(true_pixels[c], pred_pixels[c], eps=eps)
        if not np.isnan(cc_c):
            cc_list.append(cc_c)
    cc = np.mean(cc_list) if len(cc_list) > 0 else np.nan

    # ---- SAM（逐像元光谱角，再取平均）----
    # true_pixels / pred_pixels: (C, N)
    dot = np.sum(true_pixels * pred_pixels, axis=0)
    norm_true = np.linalg.norm(true_pixels, axis=0)
    norm_pred = np.linalg.norm(pred_pixels, axis=0)
    denom = np.maximum(norm_true * norm_pred, eps)

    cos_theta = np.clip(dot / denom, -1.0, 1.0)
    sam = np.degrees(np.mean(np.arccos(cos_theta)))  # 转成角度，更直观

    return {
        'Bias': bias,
        'RMSE': rmse,
        'CC': cc,
        'SAM': sam
    }


def find_file_by_base(directory, base_name):
    """
    在目录中查找 base_name.tif / base_name.tiff
    """
    for ext in ['.tif', '.tiff']:
        f = os.path.join(directory, base_name + ext)
        if os.path.exists(f):
            return f
    return None

def find_pred_file_fuzzy(pred_dir, true_base_name, replace_rule=('_L_', '_save_img_')):
    # 1) 精确替换后查找
    pred_base = build_pred_base_name(true_base_name, replace_rule)
    f = find_file_by_base(pred_dir, pred_base)
    if f:
        return f

    # 2) 提取坐标片段（最后4段或从第4段开始），按通配符查找
    parts = true_base_name.split('_')
    coords = '_'.join(parts[3:]) if len(parts) >= 4 else None
    patterns = []
    if coords:
        patterns += [f"*{coords}.tif", f"*{coords}.tiff",
                     f"*{coords}*save_img*.tif", f"*{coords}*save_img*.tiff",
                     f"*{coords}*show_fine_img*.png", f"*{coords}*show_fine_img*.tif"]

    # 3) 退化策略：查找包含整个 true_base_name 的文件（带前缀或后缀）
    patterns += [f"*{true_base_name}*.tif", f"*{true_base_name}*.tiff", f"*{true_base_name}*.png"]

    # 4) 遍历模式，返回第一个匹配项
    for pat in patterns:
        matches = glob.glob(os.path.join(pred_dir, pat))
        if matches:
            return matches[0]

    return None


def build_pred_base_name(true_base_name, replace_rule=('_L_', '_save_img_')):
    """
    根据真实文件名构造预测文件名
    """
    src_token, dst_token = replace_rule
    return true_base_name.replace(src_token, dst_token)


def build_cdl_base_name(true_base_name, year):
    """
    根据真实文件名和解析出的年份构造 CDL patch 文件名
    """
    if year is None:
        return true_base_name
        
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
        dataset_split.append({
            'group_id': group_id,
            'year': get_year_from_filename(target_date) or target_date.split('-')[0],
        })
    return dataset_split


# ================= 主流程 =================
def main():
    # ========= 路径配置 =========
    LOCATION = 'ML'    
    SPLIT = 'test'      
    
    true_dir = 'data/spatio_temporal_fusion/ML/private_data/syy_setting-9/test/patch/Landsat_02'
    cdl_dir = 'data/spatio_temporal_fusion/ML/CDL_patch'
    out_dir = 'PPT/box_plot_cropwise_4'
    os.makedirs(out_dir, exist_ok=True)

    pred_dirs_dict = {
        'STARFM': 'results/starfm/syy_setting~9/ML/one_pair~patch_size_120~patch_stride_50~window_size_51~num_classes_20/patch/imgs/ML/save_img',
        'STFGAN': 'results/stfgan/syy_setting-9/ML/inference~stage_1~RMSProp~stage_2~RMSProp/patch/imgs/stage_2/ML/save_img',
        'SwinSTFM': 'results/swinstf/syy_setting-9/ML/inference~Adam_1e-4~StepLR_step_size-15_gamma_5e-1/patch/imgs/ML/save_img',
        'STFMamba': 'results/stfmamba/syy_setting-9/ML/inferencer/patch/imgs/ML/save_img',
        'STFDiff': 'results/stfdiff/syy_setting-9/model6_GN_SiLU/ML/inference/patch/imgs/ML/0/save_img',
        'LapSTFDiff': 'results/LapSTFDiff/syy_setting-9/ML/inference/patch/imgs/ML/0/save_img',
    }

    # 指定你希望在图中突出的核心模型名称（与 pred_dirs_dict 中的键名一致）
    target_model = 'LapSTFDiff' 

    pred_replace_rule = ('_L_', '_save_img_')

    dataset_split = get_inference_config(LOCATION, split=SPLIT)
    group_to_year = {item['group_id']: item['year'] for item in dataset_split}
    
    if not group_to_year:
        print("⚠️ 警告：从 DATASET_SETTING 获取年份失败，使用内置 fallback 映射...")
        group_to_year = {
            'Group_01': '2021',
            'Group_02': '2022',
            'Group_03': '2023',
            'Group_04': '2023',
        }
    print(f"当前 Group 解析映射关系为: {group_to_year}")

    # ========= 作物类别映射 =========
    crop_value_map = {
        'Corn': [1],
        'Soybean': [5],
        'Grassland': [176],
        'Forest': [141, 143],
        'Developed': [121, 122, 123, 124]
    }

    # ========= 统计参数 =========
    metrics_to_plot = ['RMSE', 'CC', 'SAM']
    min_pixels_per_crop = 30      
    min_ratio_per_crop = 0.05
    skip_all_zero_true_patch = True
    eps = 1e-8

    methods = list(pred_dirs_dict.keys())
    crops = list(crop_value_map.keys())

    results = {
        crop: {
            metric: {method: [] for method in methods}
            for metric in metrics_to_plot
        }
        for crop in crops
    }

    valid_extensions = ('.tif', '.tiff')
    true_files = [
        f for f in glob.glob(os.path.join(true_dir, '*'))
        if f.lower().endswith(valid_extensions)
    ]

    print(f'共发现 {len(true_files)} 张真实 patch，开始进行 crop-wise 统计...\n')

    n_total = 0
    n_used = 0

    for true_file in true_files:
        n_total += 1
        true_base_name = os.path.splitext(os.path.basename(true_file))[0]

        match_group = re.search(r'(Group_\d+)', true_base_name)
        group_id = match_group.group(1) if match_group else None
        year = group_to_year.get(group_id)

        cdl_base_name = build_cdl_base_name(true_base_name, year)
        cdl_file = find_file_by_base(cdl_dir, cdl_base_name)
        
        if cdl_file is None:
            continue

        try:
            true_img = preprocess_image(tiff.imread(true_file))
            cdl_mask = preprocess_mask(tiff.imread(cdl_file))
        except Exception:
            continue

        if true_img.shape[1:] != cdl_mask.shape:
            continue

        if skip_all_zero_true_patch and np.max(true_img) <= 1e-5:
            continue

        missing_any_method = False
        valid_pred_imgs = {}
        for method in methods:
            pred_dir = pred_dirs_dict[method]
            pred_file = find_pred_file_fuzzy(pred_dir, true_base_name, pred_replace_rule)
            if pred_file is None:
                missing_any_method = True
                break
                
            try:
                pred_img = preprocess_image(tiff.imread(pred_file))
                if pred_img.shape != true_img.shape:
                    missing_any_method = True
                    break
                valid_pred_imgs[method] = pred_img
            except Exception:
                missing_any_method = True
                break

        if missing_any_method:
            continue

        patch_has_any_valid_crop = False

        for crop in crops:
            crop_codes = crop_value_map[crop]
            crop_mask = np.isin(cdl_mask, crop_codes)

            n_crop_pixels = int(np.sum(crop_mask))
            H, W = cdl_mask.shape
            
            threshold = max(min_pixels_per_crop, min_ratio_per_crop * H * W)
            if n_crop_pixels < threshold:
                continue

            crop_has_data_in_this_patch = False
            temp_results = {m_name: {} for m_name in metrics_to_plot}
            
            for method in methods:
                pred_img = valid_pred_imgs[method]
                metric_dict = compute_masked_metrics(true_img, pred_img, crop_mask, eps=eps)
                if metric_dict is not None and not np.isnan(metric_dict['CC']):
                    for metric in metrics_to_plot:
                        temp_results[metric][method] = metric_dict[metric]
                    crop_has_data_in_this_patch = True
                else:
                    crop_has_data_in_this_patch = False
                    break 

            if crop_has_data_in_this_patch:
                for metric in metrics_to_plot:
                    for method in methods:
                        results[crop][metric][method].append(temp_results[metric][method])
                patch_has_any_valid_crop = True

        if patch_has_any_valid_crop:
            n_used += 1

    print(f'统计完成：总 patch={n_total}, 有效参与统计 patch={n_used}')

    # ================= 绘图优化部分 =================
    # print('开始绘制 crop-wise 箱线图...')

    nrows = len(crops)
    ncols = len(metrics_to_plot)

    # 1. 保持 methods 列表为字典原始键，用于提取数据
    methods = ['STARFM', 'SwinSTFM', 'STFGAN', 'STFMamba', 'STFDiff', 'LapSTFDiff']
    
    # 定义用于显示的带有换行的标签列表
    display_methods = ['STARFM', 'Swin\nSTFM', 'STFGAN', 'STF\nMamba', 'STFDiff', 'Lap\nSTFDiff']
    
    # 2. 完美复刻的高级学术配色
    color_baseline = '#B0C4DE'  # 莫兰迪灰蓝 (贴近你的参考图)
    color_target = '#C00000'    # 经典深红，用于突出目标模型
    box_colors = [color_target if m == target_model else color_baseline for m in methods]

    # 3. 核心改动：用紧凑的宽度并利用换行解决重叠
    # 将宽度调回较紧凑的 15 或 16
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(18, 11),  
        sharex=True,
        gridspec_kw={
            'hspace': 0.05,  # 极小化上下子图间距
            'wspace': 0.15   # 左右子图间距
        }
    )

    for r, crop in enumerate(crops):
        for c, metric_name in enumerate(metrics_to_plot):
            ax = axes[r, c]

            data_to_plot = []
            for method in methods:
                # 使用原始的 method 名字去取数据，避免 KeyError
                vals = results[crop][metric_name][method]
                vals = [v for v in vals if not np.isnan(v)]
                if len(vals) == 0:
                    data_to_plot.append([0.0])
                else:
                    data_to_plot.append(vals)

            # 4. 箱体宽度配合紧凑的画布
            bplot = ax.boxplot(
                data_to_plot,
                patch_artist=True,
                widths=0.45,
                showfliers=False,
                boxprops=dict(color='#333333', linewidth=1.0),
                medianprops=dict(color='white', linewidth=1.8),
                whiskerprops=dict(color='#333333', linewidth=1.0),
                capprops=dict(color='#333333', linewidth=1.0)
            )

            # 填充颜色
            for patch, color in zip(bplot['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(1.0)

            # --- 标题与 Y 轴标签 ---
            if r == 0:
                # ax.set_title(metric_name, fontsize=18, fontweight='bold', pad=12)
                ax.set_title(metric_name, fontsize=18, pad=12)

            if c == 0:
                # ax.set_ylabel(crop, fontsize=16, fontweight='bold')
                ax.set_ylabel(crop, fontsize=16,)
            else:
                ax.set_ylabel('')

            # --- Y 轴刻度与网格优化 ---
            ax.tick_params(axis='y', labelsize=12)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.grid(True, linestyle='--', alpha=0.6, color='#BBBBBB')
            ax.set_axisbelow(True)

            # 去除顶部和右侧边框
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # 隐藏内部所有 X 轴标签
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # 5. 底部 X 轴标签单独处理 (使用带换行的名字)
    for c in range(ncols):
        ax_bottom = axes[nrows - 1, c]
        ax_bottom.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
        ax_bottom.set_xticks(range(1, len(display_methods) + 1))
        # 应用带换行符的 display_methods
        ax_bottom.set_xticklabels(display_methods, rotation=0, ha='center', fontsize=13)

    # 使用 subplots_adjust 精确锁定布局，由于有两行文字，底部空间稍微给多一点 
    plt.subplots_adjust(top=0.92, bottom=0.1, left=0.08, right=0.98)
    
    save_path = os.path.join(out_dir, 'CropWise_BoxPlot_Metrics.png')
    plt.savefig(save_path, dpi=400, facecolor='white')
    
    save_path = os.path.join(out_dir, 'CropWise_BoxPlot_Metrics.pdf')
    plt.savefig(save_path, facecolor='white', bbox_inches='tight')
    plt.close()

    print(f'箱线图已保存至: {save_path}')
if __name__ == '__main__':
    main()