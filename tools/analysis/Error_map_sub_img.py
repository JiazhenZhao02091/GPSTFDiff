"""
功能:
    按 Group 同时生成 GT/预测影像的 321 RGB 图、误差图和对应裁剪 patch，并为每组保存统一色标。

参数:
    true_dir_dict (dict[str, str]): 真实影像名称到目录的映射。
    pred_dirs_dict (dict[str, str]): 模型名称到预测影像目录的映射。
    output_dir (str): 输出根目录。
    bboxes (list[list[int]]): 裁剪框列表，格式为 [y_start, y_end, x_start, x_end]。
    cmap (str): 误差图使用的 matplotlib 色带名称。

返回:
    None。函数直接保存图片文件。

输出:
    输出 full_321、full_errormap、patch_321、patch_errormap 和 colorbar PNG 文件。
"""
import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.cm as cm
from PIL import Image, ImageDraw

# 假设这两个模块在您的项目中可用，若报错请确保路径正确
from src.utils.img.process.linear_stretch import truncated_linear_stretch
from scripts.spatio_temparol_fusion.format import format_data

# ==========================================
# 顶刊图表全局字体设置
# ==========================================
font_path = os.path.expanduser('~/.local/share/fonts/times.ttf')
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.unicode_minus'] = False

def preprocess_to_reflectance(img_array):
    """将遥感图像还原至 0-1 的反射率"""
    if np.max(img_array) > 1.5:
        return img_array / 10000.0
    return img_array

def get_rgb_321(img_raw):
    """
    针对 0-10000 或 0-1 量纲进行自适应处理
    """
    # 1. 强制转为 float32 并格式化
    img_raw = img_raw.astype(np.float32)
    img_fmt = format_data(img_raw, 6)
    
    # 2. 提取 RGB 波段
    if img_fmt.shape[2] >= 4:
        rgb_raw = img_fmt[:, :, (3, 2, 1)]
    else:
        rgb_raw = img_fmt

    # 3. 【核心修复】自适应量纲恢复
    # 如果最大值很小，说明已经是反射率，放大到 0-10000 保证线性拉伸的数值稳定性
    curr_max = np.max(rgb_raw)
    if curr_max < 2.0:
        rgb_raw = rgb_raw * 10000.0
        # print(f"DEBUG: 检测到反射率量纲，已放大。新 Max: {np.max(rgb_raw)}")

    # 4. 调用带保护的拉伸函数
    image = safe_truncated_linear_stretch(rgb_raw)
    return image

def safe_truncated_linear_stretch(image, truncated_percent=2):
    """
    增加了除零保护和类型转换保护的拉伸函数
    """
    # 计算分位数
    low = np.percentile(image, truncated_percent, axis=(0, 1), keepdims=True)
    high = np.percentile(image, 100 - truncated_percent, axis=(0, 1), keepdims=True)

    # 【数值保护】防止分母为 0
    diff = high - low
    diff = np.where(diff <= 0, 1.0, diff) 

    # 线性拉伸
    stretched = (image - low) / diff * 255.0
    
    # 【类型保护】先剪裁，后转码，防止溢出或 NaN 导致全黑
    stretched = np.nan_to_num(stretched) # 将 NaN 转为 0
    stretched = np.clip(stretched, 0, 255)
    
    return stretched.astype(np.uint8)

def process_combined_images(true_dir_dict, pred_dirs_dict, output_dir, bboxes, cmap='OrRd'):
    """
    核心整合逻辑：按 Group 读取，统一计算色标上限，裁剪并保存 ErrorMap 和 321 RGB 图
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 提取 GT 目录并寻找所有的 Group 分组
    gt_name = list(true_dir_dict.keys())[0]
    gt_dir = true_dir_dict[gt_name]
    valid_exts = ('.tif', '.tiff')
    
    gt_files = [f for f in os.listdir(gt_dir) if f.lower().endswith(valid_exts)]
    
    # 提取文件名前缀，如 Group_01, Group_02
    group_prefixes = set()
    for f in gt_files:
        if f.startswith('Group_'):
            parts = f.split('_')
            if len(parts) >= 2:
                group_prefixes.add(f"{parts[0]}_{parts[1]}")
    
    print(f"发现以下分组: {sorted(list(group_prefixes))}\n" + "="*50)

    for group in sorted(group_prefixes):
        print(f"🚀 正在处理分组: {group} ...")
        
        # 为当前 Group 设立专属文件夹
        group_out_dir = os.path.join(output_dir, group)
        os.makedirs(group_out_dir, exist_ok=True)
        
        # 读取该组的 GT 图像
        gt_filename = next((f for f in gt_files if f.startswith(group)), None)
        if not gt_filename: 
            continue
            
        gt_path = os.path.join(gt_dir, gt_filename)
        try:
            gt_raw = tiff.imread(gt_path).astype(np.float32)
            gt_ref = preprocess_to_reflectance(gt_raw)
        except Exception as e:
            print(f"读取 GT 失败: {gt_path}, {e}")
            continue

        # ---------------------------------------------------------
        # 第一阶段：遍历所有模型，读取数据并计算 Error，寻找全局 Colorbar 上限
        # ---------------------------------------------------------
        valid_preds = {}
        scene_vmax = 0.0
        
        for model_name, pred_dir in pred_dirs_dict.items():
            if not os.path.exists(pred_dir): continue
            
            pred_files = [f for f in os.listdir(pred_dir) if f.lower().endswith(valid_exts)]
            pred_filename = next((f for f in pred_files if f.startswith(group)), None)
            
            if not pred_filename: continue
            
            pred_path = os.path.join(pred_dir, pred_filename)
            try:
                pred_raw = tiff.imread(pred_path).astype(np.float32)
                pred_ref = preprocess_to_reflectance(pred_raw)
            except Exception:
                continue

            if gt_ref.shape != pred_ref.shape: 
                print(f"   ⚠️ 模型 {model_name} 形状不匹配，跳过。")
                continue
            
            # 计算绝对误差并按波段平均
            error = np.abs(gt_ref - pred_ref)
            if error.ndim == 3:
                error = np.mean(error, axis=0) if error.shape[0] < error.shape[-1] else np.mean(error, axis=-1)
                
            valid_preds[model_name] = (pred_raw, error)
            
            # 计算 98% 分位数作为该图的色条上限
            p98 = np.percentile(error, 98)
            if p98 > scene_vmax:
                scene_vmax = p98
                
        if not valid_preds:
            print(f"   ⚠️ 分组 {group} 没有找到任何匹配的预测图，跳过。")
            continue

        print(f"   📊 统一色标 (Colorbar) 上限设定为: {scene_vmax:.4f}")

        # 配置 matplotlib 颜色映射器 (带 Gamma 非线性映射)
        gamma_norm = mcolors.PowerNorm(gamma=1.5, vmin=0, vmax=scene_vmax)
        sm = cm.ScalarMappable(cmap=cmap, norm=gamma_norm)
        sm.set_array([])

        # ---------------------------------------------------------
        # 第二阶段：保存 GT 的 321 图像及 Patch
        # ---------------------------------------------------------
        gt_rgb_array = get_rgb_321(gt_raw)
        gt_rgb_img = Image.fromarray(gt_rgb_array)
        
        # 1. 保存 GT 全图
        gt_rgb_img.save(os.path.join(group_out_dir, f"{gt_name}_full_321.png"))
        
        # (可选) 保存一张带有红框的 GT 全图方便参照
        if bboxes:
            marked_img = gt_rgb_img.copy()
            draw = ImageDraw.Draw(marked_img)
            for bbox in bboxes:
                y_start, y_end, x_start, x_end = bbox
                draw.rectangle([x_start, y_start, x_end, y_end], outline='red', width=3)
            marked_img.save(os.path.join(group_out_dir, f"{gt_name}_full_marked_321.png"))

            # 2. 保存 GT 的 Patches
            for bbox in bboxes:
                y_start, y_end, x_start, x_end = bbox
                patch_str = f"{y_start}_{y_end}_{x_start}_{x_end}"
                
                gt_patch = gt_rgb_array[y_start:y_end, x_start:x_end, :]
                Image.fromarray(gt_patch).save(os.path.join(group_out_dir, f"{gt_name}_{patch_str}_patch_321.png"))

        # ---------------------------------------------------------
        # 第三阶段：保存 各模型的 321图像 和 ErrorMap (全图 + Patch)
        # ---------------------------------------------------------
        for model_name, (pred_raw, error) in valid_preds.items():
            
            # 提取预测图的 321 RGB
            pred_rgb_array = get_rgb_321(pred_raw)
            
            # 1. 保存 Full 321 和 Full ErrorMap
            Image.fromarray(pred_rgb_array).save(os.path.join(group_out_dir, f"{model_name}_full_321.png"))
            
            # 【高级技巧】：利用 sm.to_rgba 直接将误差矩阵映射为 RGB(A) 像素数组，
            # 并用 Image.fromarray 保存。速度极快，且100%无白边。
            error_rgba = sm.to_rgba(error, bytes=True) 
            Image.fromarray(error_rgba).save(os.path.join(group_out_dir, f"{model_name}_full_errormap.png"))

            # 2. 遍历保存 Patches
            if bboxes:
                for bbox in bboxes:
                    y_start, y_end, x_start, x_end = bbox
                    patch_str = f"{y_start}_{y_end}_{x_start}_{x_end}"
                    
                    # 裁剪并保存 Patch 321
                    patch_321 = pred_rgb_array[y_start:y_end, x_start:x_end, :]
                    Image.fromarray(patch_321).save(os.path.join(group_out_dir, f"{model_name}_{patch_str}_patch_321.png"))
                    
                    # 裁剪并保存 Patch ErrorMap
                    patch_error = error[y_start:y_end, x_start:x_end]
                    patch_err_rgba = sm.to_rgba(patch_error, bytes=True)
                    Image.fromarray(patch_err_rgba).save(os.path.join(group_out_dir, f"{model_name}_{patch_str}_patch_errormap.png"))

        # ---------------------------------------------------------
        # 第四阶段：保存该 Group 的通用 Colorbar 标尺
        # ---------------------------------------------------------
        fig_cbar, ax_cbar = plt.subplots(figsize=(0.6, 5))
        cbar = fig_cbar.colorbar(sm, cax=ax_cbar)
        cbar.ax.tick_params(labelsize=14) 
        cbar_path = os.path.join(group_out_dir, f'colorbar_{group}.png')
        fig_cbar.savefig(cbar_path, bbox_inches='tight', dpi=600, facecolor='white')
        plt.close(fig_cbar)

        print(f"   ✅ 分组 {group} 处理完毕。")


if __name__ == '__main__':
    # 真实图像字典 (包含简称和路径)
    TRUE_DIR_DICT = {
        'GT' : '/home/zhaojiazhen/workspace/STF/STF/data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/test/full/Landsat_02'
    }
    # 模型预测图像字典 (包含简称和路径)
    PRED_DIRS_DICT = {
        'DDIM_step_10' : '/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/diff_step/DDIM/inference_10step/full/imgs/CIA/0/save_img',
        'DDIM_step_100' : '/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/diff_step/DDIM/inference_100step/full/imgs/CIA/0/save_img',
        'lap_step_10' : '/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/diff_step/lap/inferency_10step/full/imgs/CIA/0/save_img',
        'lap_step_100' : '/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/diff_step/lap/inferency_100step/full/imgs/CIA/0/save_img',
    }

    # 裁剪范围列表 [y_start, y_end, x_start, x_end]
    CROP_BBOXES = [
        [1320, 1576, 760, 1016],
        [820, 1076, 900, 1156]
    ]

    # 输出主目录 (脚本会自动在里面建立 Group_01, Group_02 等子文件夹)
    OUTPUT_DIR = '/home/zhaojiazhen/workspace/STF/STF/artifacts/metric_log/abl/error_map'

    process_combined_images(
        true_dir_dict=TRUE_DIR_DICT,
        pred_dirs_dict=PRED_DIRS_DICT,
        output_dir=OUTPUT_DIR,
        bboxes=CROP_BBOXES,
        cmap='viridis' # 可调整色带，如 'jet', 'viridis' 等
    )
