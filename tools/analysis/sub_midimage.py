"""
功能:
    按图片尺寸选择对应裁剪框，从中间结果 PNG/TIF 中裁剪 patch，并在原图目录保存带框整图。

参数:
    input_dir (str): 待裁剪图片目录。
    output_dir (str): 裁剪 patch 输出目录。
    crop_info (dict[str, list[int]]): 图片尺寸到裁剪框的映射，裁剪框格式为 [y_start, y_end, x_start, x_end]。

返回:
    None。函数直接保存图片文件。

输出:
    输出裁剪 patch 到 output_dir，并保存带框全图到原图片目录。
"""
import os
from PIL import Image, ImageDraw

# CROP_BBOXES = {
#     '1280': [900, 1124, 666, 891],
#     '640': [450, 562, 333, 445],
#     '320': [225, 281, 167, 223]
# }

CROP_BBOXES = {
    '1280': [461, 686, 789, 1013],
    '640': [231, 343, 395, 507],
    '320': [115, 171, 197, 253]
}

def crop_images_by_size(input_dir, output_dir, crop_info):
    os.makedirs(output_dir, exist_ok=True)
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    filenames = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    print(f"共发现 {len(filenames)} 张图片，开始裁剪...\n")
    for filename in filenames:
        file_path = os.path.join(input_dir, filename)
        try:
            img = Image.open(file_path).convert("RGB")
            w, h = img.size
            # 只处理正方形
            if w != h:
                print(f"跳过非正方形图片: {filename}")
                continue
            key = str(w)
            if key not in crop_info:
                print(f"未找到尺寸 {w} 的裁剪参数，跳过: {filename}")
                continue
            y_start, y_end, x_start, x_end = crop_info[key]

            # 裁剪并保存 patch 到 output_dir
            cropped_img = img.crop((x_start, y_start, x_end, y_end))
            base_name = os.path.splitext(filename)[0]
            patch_name = f"{base_name}_{y_start}_{y_end}_{x_start}_{x_end}.png"
            patch_path = os.path.join(output_dir, patch_name)
            cropped_img.save(patch_path)
            print(f"✅ 已保存 patch: {patch_name}")

            # 在原图上绘制矩形并保存到原文件夹（命名参考 patch）
            try:
                annotated = img.copy()
                draw = ImageDraw.Draw(annotated)
                # rectangle expects (left, top, right, bottom) -> (x_start, y_start, x_end, y_end)
                draw.rectangle([x_start, y_start, x_end, y_end], outline='red', width=3)
                orig_dir = os.path.dirname(file_path)
                annotated_name = f"{base_name}_{y_start}_{y_end}_{x_start}_{x_end}.png"
                annotated_path = os.path.join(orig_dir, annotated_name)
                annotated.save(annotated_path)
                print(f"✅ 已保存带框整图: {annotated_name}（保存在原文件夹）")
            except Exception as e:
                print(f"⚠️ 保存带框整图失败: {e}")

        except Exception as e:
            print(f"❌ 处理图片 {filename} 时出错: {e}")
    print("\n🎉 所有裁剪完成！")

# 用法示例（修改为实际路径后运行）
if __name__ == "__main__":
    input_dir = "/home/zhaojiazhen/workspace/STF/STF/artifacts/image/midimage/origin/png"
    output_dir = "/home/zhaojiazhen/workspace/STF/STF/artifacts/image/midimage/origin/png_patch2"
    crop_images_by_size(input_dir, output_dir, CROP_BBOXES)
