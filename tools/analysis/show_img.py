"""
功能:
    批量把多波段 tif/tiff 影像格式化、线性拉伸为 RGB，并居中裁剪为正方形 PNG。

参数:
    src_dir (str): 源 tif/tiff 目录。
    res_dir (str): PNG 输出目录。

返回:
    None。函数直接保存图片文件。

输出:
    在输出目录保存同名 .png 文件，并打印每张图的保存路径和形状。
"""
import os
import numpy as np
import tifffile as tiff
from PIL import Image
from src.utils.img.process.linear_stretch import truncated_linear_stretch
from scripts.spatio_temparol_fusion.format import format_data

def batch_convert_tif_to_png(src_dir, res_dir):
    """
    批量读取 src_dir 下所有 tif/tiff 文件，拉伸处理后保存为 png 到 res_dir。
    文件名与原文件名一致（后缀改为 .png）。
    """
    os.makedirs(res_dir, exist_ok=True)
    tif_files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.tif', '.tiff'))]
    print(f"共发现 {len(tif_files)} 个 tif 文件。")

    for fname in tif_files:
        tif_path = os.path.join(src_dir, fname)
        img = tiff.imread(tif_path)
        img = format_data(img, 6)
        stretched_img = truncated_linear_stretch(img, stretch_range=[0, 255])
        image = stretched_img.astype(np.uint8)
        # 选取 RGB 波段
        if image.shape[2] >= 4:
            image = image[:, :, (2, 1, 0)]
        # 裁剪为正方形
        h, w = image.shape[:2]
        side = min(h, w)
        y_start = (h - side) // 2
        x_start = (w - side) // 2
        image_square = image[y_start:y_start+side, x_start:x_start+side, :]
        png_name = os.path.splitext(fname)[0] + ".png"
        png_path = os.path.join(res_dir, png_name)
        img_pil = Image.fromarray(image_square)
        img_pil.save(png_path)
        print(f"✅ 已保存: {png_path}，形状为: {image_square.shape}")

# 用法示例
src_dir = "/home/zhaojiazhen/workspace/STF/STF/artifacts/image/midimage/origin/lap"
res_dir = "/home/zhaojiazhen/workspace/STF/STF/artifacts/image/midimage/origin/png"
batch_convert_tif_to_png(src_dir, res_dir)
