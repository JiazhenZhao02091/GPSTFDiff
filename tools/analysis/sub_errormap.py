import os
import argparse
from PIL import Image, ImageDraw

def crop_errormaps_from_subdirs(input_dir, output_dir, bboxes):
    """
    遍历输入目录下的所有子文件夹（模型名），读取其中的图片，
    根据传入的 bboxes 列表进行裁剪，并把结果保存在输出目录，
    文件名前缀加上模型名。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    
    # 获取 input_dir 下的所有子项
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    print(f"🔍 查找到 {len(subdirs)} 个模型子文件夹，开始裁剪...\n")
    
    for model_name in subdirs:
        model_dir = os.path.join(input_dir, model_name)
        
        # 遍历子文件夹中的图像文件
        filenames = [f for f in os.listdir(model_dir) if f.lower().endswith(valid_extensions)]
        if not filenames:
            continue
            
        print(f"📦 正在处理模型: [{model_name}]，共 {len(filenames)} 张图片")
        
        for filename in filenames:
            file_path = os.path.join(model_dir, filename)
            base_name = os.path.splitext(filename)[0]  # 原文件名去除后缀
            
            try:
                # 读取图像 (ErrorMap 通常是 RGB 或 RGBA 的 PNG 图像)
                img = Image.open(file_path).convert("RGBA")
                
                # 创建用于绘制全局标定框的图像副本
                img_with_boxes = img.copy().convert("RGB")
                draw = ImageDraw.Draw(img_with_boxes)
                
                # 遍历用户提供的每个剪切框
                for bbox in bboxes:
                    y_start, y_end, x_start, x_end = bbox
                    
                    # 注意：PIL 图像的 crop 期望的坐标顺序为 (left, upper, right, lower)
                    # 对应于您的矩阵切片习惯，应该是 (x_start, y_start, x_end, y_end)
                    cropped_img = img.crop((x_start, y_start, x_end, y_end))
                    
                    # 构造新的文件名：模型名_原文件名_坐标范围.png
                    # 比如：starfm_errormap_Landsat_02_1320_1576_760_1016.png
                    patch_name = f"{model_name}_{base_name}_{y_start}_{y_end}_{x_start}_{x_end}.png"
                    patch_path = os.path.join(output_dir, patch_name)
                    
                    cropped_img.save(patch_path)
                    print(f"   ✅ 已保存: {patch_name}")
                    
                    # 在全图中绘制红色矩形框 (left, upper, right, lower)
                    draw.rectangle([x_start, y_start, x_end, y_end], outline="red", width=3)
                
                # 保存带有所有标定框的完整图像
                full_img_name = f"{model_name}_{base_name}_full_with_boxes.png"
                full_img_path = os.path.join(output_dir, full_img_name)
                img_with_boxes.save(full_img_path)
                print(f"   ✅ 已保存全局框选图: {full_img_name}")
                    
            except Exception as e:
                print(f"   ❌ 读取或处理图片时发生错误 {file_path}: {e}")
                
    print("\n🎉 所有子图裁剪完成！\n" + "-"*40)


if __name__ == '__main__':
    # 裁剪范围 [y_start, y_end, x_start, x_end]
    # CIA
    # CROP_BBOX_1 = [1320, 1576, 760, 1016]
    # CROP_BBOX_2 = [820, 1076, 900, 1156]
    # # 将多个边框组成一个列表
    # CROP_BBOXES = [CROP_BBOX_1, CROP_BBOX_2]

    # CROP_BBOX_1 = [120, 376, 140, 396]
    # CROP_BBOX_2 = [700, 956, 980, 1236]
    # # 将多个边框组成一个列表
    # CROP_BBOXES = [CROP_BBOX_1, CROP_BBOX_2]
    
    # ML
    # CROP_BBOX_1 =[676, 932, 608, 864]
    # CROP_BBOX_2 = [206, 462, 1008, 1264]
    CROP_BBOX_1 =[676, 932, 608, 864]
    CROP_BBOX_2 = [206, 462, 1008, 1264]
    # 将多个边框组成一个列表
    CROP_BBOXES = [CROP_BBOX_1, CROP_BBOX_2]

    parser = argparse.ArgumentParser(description='Crop sub-images from directory containing model sub-folders.')
    
    # 这里您可以根据实际路径修改默认值，或者在终端通过命令行传入
    parser.add_argument('--input_dir', type=str, 
                        default='PPT/Errormap/ML/individual', 
                        help='包含多个模型子文件夹的基础目录')
    parser.add_argument('--output_dir', type=str, 
                        default='PPT/Errormap/ML/individual_sub_boxes', 
                        help='保存裁剪后图像的目标目录')
    
    args = parser.parse_args()

    print(f"输入路径: {args.input_dir}")
    print(f"输出路径: {args.output_dir}\n")

    crop_errormaps_from_subdirs(
        input_dir=args.input_dir, 
        output_dir=args.output_dir, 
        bboxes=CROP_BBOXES
    )