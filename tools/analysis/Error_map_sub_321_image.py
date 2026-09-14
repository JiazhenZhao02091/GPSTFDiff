"""
功能:
    将 CIA 全图及预测结果转换为 3-2-1 波段 RGB 图，并按 Group 保存全图和指定区域 patch。

参数:
    TRUE_IMAGES_DIR (str): 真实影像目录。
    PRED_DIRS_DICT (dict[str, str]): 模型名称到预测影像目录的映射。
    CROP_BBOXES (list[list[int]]): 裁剪框列表，格式为 [y_start, y_end, x_start, x_end]。
    OUTPUT_ROOT (str): 输出根目录。

返回:
    None。脚本通过保存图片输出结果。

输出:
    在 OUTPUT_ROOT/Group_* 下保存 GT 和各模型的 full_321 与 patch_321 PNG 文件。
"""
import os
import numpy as np
import tifffile as tiff
from PIL import Image, ImageDraw
import argparse

# 导入你项目中的工具函数，请确保路径正确
from src.utils.img.process.linear_stretch import truncated_linear_stretch
from scripts.spatio_temparol_fusion.format import format_data

def get_rgb_321(img_raw):
    """
    核心转换逻辑：格式化 -> 量纲对齐 -> 提取 (2,1,0) -> 线性拉伸
    """
    # 1. 强制转为 float32 确保计算不溢出
    img_raw = img_raw.astype(np.float32)
    
    # 2. 格式化为 HWC 结构 (假设为 6 波段)
    img_fmt = format_data(img_raw, 6)
    
    # 3. 自适应量纲恢复：如果最大值很小(反射率)，放大到 10000
    # 这是防止线性拉伸函数因数值太小产生除零 NaN 的关键 
    if np.max(img_fmt) < 1.5:
        img_fmt = img_fmt * 10000.0
    
    # 4. 根据你的成功参考代码，提取波段 (2, 1, 0)
    if img_fmt.shape[2] >= 3:
        rgb_raw = img_fmt[:, :, (2, 1, 0)]
    else:
        rgb_raw = img_fmt

    # 5. 执行 2% 线性拉伸到 0-255
    # 注意：确保你的 truncated_linear_stretch 内部不包含将数据除以 10000 的预处理
    image_stretched = truncated_linear_stretch(rgb_raw, stretch_range=[0, 255])
    
    # 6. 强制转换为 uint8
    return image_stretched.astype(np.uint8)

def process_321_batch(true_dir, pred_dirs_dict, output_root, bboxes):
    """
    批量处理逻辑：按 Group 分类存储
    """
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    valid_exts = ('.tif', '.tiff')
    gt_files = [f for f in os.listdir(true_dir) if f.lower().endswith(valid_exts)]
    
    # 提取所有 Group 前缀
    groups = sorted(list(set(['_'.join(f.split('_')[:2]) for f in gt_files if f.startswith('Group_')])))
    
    print(f"找到分组: {groups}")

    for group in groups:
        print(f"🚀 正在处理分组: {group} ...")
        group_dir = os.path.join(output_root, group)
        os.makedirs(group_dir, exist_ok=True)

        # 1. 处理 GT
        gt_file = next((f for f in gt_files if f.startswith(group)), None)
        if gt_file:
            gt_path = os.path.join(true_dir, gt_file)
            img_rgb = get_rgb_321(tiff.imread(gt_path))
            save_full_and_patches(img_rgb, group_dir, "GT", bboxes)

        # 2. 处理各个模型预测图
        for model_name, pred_dir in pred_dirs_dict.items():
            if not os.path.exists(pred_dir): continue
            
            p_files = [f for f in os.listdir(pred_dir) if f.lower().endswith(valid_exts)]
            p_file = next((f for f in p_files if f.startswith(group)), None)
            
            if p_file:
                p_path = os.path.join(pred_dir, p_file)
                try:
                    p_rgb = get_rgb_321(tiff.imread(p_path))
                    save_full_and_patches(p_rgb, group_dir, model_name, bboxes)
                except Exception as e:
                    print(f"   ⚠️ 模型 {model_name} 处理失败: {e}")

def save_full_and_patches(img_rgb, save_dir, prefix, bboxes):
    """保存全图和裁剪的 Patch"""
    # 保存全图
    full_pil = Image.fromarray(img_rgb)
    full_pil.save(os.path.join(save_dir, f"{prefix}_full_321.png"))

    # 保存 Patch
    if bboxes:
        for i, bbox in enumerate(bboxes):
            y_s, y_e, x_s, x_e = bbox
            patch = img_rgb[y_s:y_e, x_s:x_e, :]
            patch_pil = Image.fromarray(patch)
            patch_pil.save(os.path.join(save_dir, f"{prefix}_patch{i}_{y_s}_{x_s}_321.png"))

if __name__ == '__main__':
    # 配置区
    TRUE_IMAGES_DIR = '/home/zhaojiazhen/workspace/STF/STF/data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/test/full/Landsat_02'
    
    PRED_DIRS_DICT = {
        'DDIM_step_10' : '/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/diff_step/DDIM/inference_10step/full/imgs/CIA/0/save_img',
        'DDIM_step_100' : '/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/diff_step/DDIM/inference_100step/full/imgs/CIA/0/save_img',
        'lap_step_10' : '/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/diff_step/lap/inferency_10step/full/imgs/CIA/0/save_img',
        'lap_step_100' : '/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/diff_step/lap/inferency_100step/full/imgs/CIA/0/save_img',
    }

    CROP_BBOXES = [
        [1320, 1576, 760, 1016],
        [820, 1076, 900, 1156]
    ]

    OUTPUT_ROOT = '/home/zhaojiazhen/workspace/STF/STF/artifacts/metric_log/abl/subimage_321'

    process_321_batch(TRUE_IMAGES_DIR, PRED_DIRS_DICT, OUTPUT_ROOT, CROP_BBOXES)
