"""
功能:
    从真实影像和多个模型预测影像中生成 RGB 全图、带框全图以及指定区域子图。

参数:
    --dataset (str): 数据集或处理模式，可选 CIA、LGC、ML、321、single。
    --image_path (str): single 模式下待裁剪的单张影像路径。
    --output_dir (str): single 模式下的输出目录。
    CROP_BBOXES (list[list[int]]): 批处理模式使用的裁剪框，格式为 [y_start, y_end, x_start, x_end]。

返回:
    None。脚本通过文件系统输出结果，不返回数据对象。

输出:
    在输出目录写入 full、full_marked 和 patch PNG 文件，并在终端打印处理进度。
"""
import os
import numpy as np
import tifffile as tiff
from PIL import Image, ImageDraw
from src.utils.img.process.linear_stretch import truncated_linear_stretch
from scripts.spatio_temparol_fusion.format import format_data
import argparse

def process_and_save_image(tif_path, save_dir, prefix_name, bboxes):
    """
    处理单张图片：格式化、拉伸、提取 RGB 波段并保存全图和裁剪的多个子图
    """
    try:
        img = tiff.imread(tif_path)
    except Exception as e:
        print(f"读取图片失败: {tif_path}，错误: {e}")
        return

    # 格式化数据，假设为 6 波段
    img = format_data(img, 6)
    
    # 线性拉伸到 0-255，并转换数据类型
    stretched_img = truncated_linear_stretch(img, stretch_range=[0, 255])
    image = stretched_img.astype(np.uint8)
    
    # 提取 RGB 通道 (假设 6 波段顺序对应的 RGB 为 3, 2, 1)
    if image.shape[2] >= 4:
        image = image[:, :, (3, 2, 1)]
    
    # 1. 保存完整图片
    full_img_pil = Image.fromarray(image)
    full_name = f"{prefix_name}_full.png"
    full_path = os.path.join(save_dir, full_name)
    full_img_pil.save(full_path)

    # 新增：保存带标记的 full 图
    if bboxes is not None:
        marked_img = full_img_pil.copy()
        draw = ImageDraw.Draw(marked_img)
        for bbox in bboxes:
            y_start, y_end, x_start, x_end = bbox
            # PIL 的坐标是 (x1, y1, x2, y2)
            draw.rectangle([x_start, y_start, x_end, y_end], outline='red', width=3)
        marked_name = f"{prefix_name}_full_marked.png"
        marked_path = os.path.join(save_dir, marked_name)
        marked_img.save(marked_path)
    
    # 2. 遍历所有的 bbox 列表截取对应的多张 patch 并保存
    if bboxes is not None:
        for bbox in bboxes:
            y_start, y_end, x_start, x_end = bbox
            patch_array = image[y_start:y_end, x_start:x_end, :]
            
            patch_img_pil = Image.fromarray(patch_array)
            
            # 将四个指标 (坐标值) 添加至文件名
            patch_name = f"{prefix_name}_{y_start}_{y_end}_{x_start}_{x_end}.png"
            patch_path = os.path.join(save_dir, patch_name)
            patch_img_pil.save(patch_path)
            
            print(f"✅ 已保存: {patch_name} (全图形状: {image.shape}, 子图形状: {patch_array.shape})")

def generate_sub_images(true_dir, pred_dirs_dict, output_dir, bboxes):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_extensions = ('.tif', '.tiff', '.png', '.jpg') 
    filenames = [f for f in os.listdir(true_dir) if f.lower().endswith(valid_extensions)]
    
    print(f"共发现 {len(filenames)} 张真实图像，开始生成子图...\n")

    for filename in filenames:
        true_path = os.path.join(true_dir, filename)
        
        # 提取文件名前缀，例如将 Landsat_02_xxxx.tif 提取为 Landsat_02
        keyword = os.path.splitext(filename)[0]
        keyword_list = keyword.split('_')
        keyword = '_'.join(keyword_list[:2])
        
        print(f"[{keyword}] 开始处理...")
        
        # 为当前图像组创建一个子文件夹以存放各个模型的结果
        scene_out_dir = os.path.join(output_dir, keyword)
        if not os.path.exists(scene_out_dir):
            os.makedirs(scene_out_dir)

        # 1. 处理真实图像 (Ground Truth) 传入 bboxes
        process_and_save_image(true_path, scene_out_dir, "GT", bboxes)

        # 2. 处理各模型的预测图像 传入 bboxes
        for model_name, pred_dir in pred_dirs_dict.items():
            if not os.path.exists(pred_dir): 
                continue
                
            available_files = [f for f in os.listdir(pred_dir) if f.lower().endswith(valid_extensions)]
            matched_pred_file = next((f for f in available_files if keyword in f), None)
            
            if not matched_pred_file: 
                print(f"   ⚠️ 模型 {model_name} 中未找到匹配 {keyword} 的数据。")
                continue
                
            pred_path = os.path.join(pred_dir, matched_pred_file)
            process_and_save_image(pred_path, scene_out_dir, model_name, bboxes)
            
        print(f"[{keyword}] 处理完毕。\n" + "-"*40)


if __name__ == '__main__':
    # 裁剪范围 [y_start, y_end, x_start, x_end]
   

    
    # CROP_BBOXES = None

    argparser = argparse.ArgumentParser(description='Generate error maps for spatio-temporal fusion results.')
    # ================= 1. 修改 choices，增加 'single' =================
    argparser.add_argument('--dataset', type=str, default='LGC', choices=['CIA', 'LGC', 'ML', 'single', '321'], help='Dataset to process (CIA, LGC, ML, or single)')
    
    # 增加为 'single' 模式提供的参数
    argparser.add_argument('--image_path', type=str, help='Path to the single image to crop (required if dataset is single)')
    argparser.add_argument('--output_dir', type=str, default='artifacts/image/subimage/single', help='Output directory for single mode')
    args = argparser.parse_args()

    # ================= 2. 增加 single 处理逻辑 =================
    if args.dataset == 'single':
        if not args.image_path:
            raise ValueError("当 --dataset 为 'single' 时，必须提供 --image_path 参数。")
        
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            
        # 提取文件名字作为前缀
        prefix_name = os.path.splitext(os.path.basename(args.image_path))[0]
        
        print(f"开始处理单张图片: {args.image_path}")
        process_and_save_image(args.image_path, args.output_dir, prefix_name, CROP_BBOXES)
        print(f"单张图片处理完毕，结果已保存在: {args.output_dir}\n" + "-"*40)
        
    else:
        # ============== 以下为原有的批量处理逻辑 ==============
        if args.dataset == 'CIA':
            CROP_BBOX_1 = [1320, 1576, 760, 1016]
            CROP_BBOX_2 = [820, 1076, 900, 1156]
            CROP_BBOXES = [CROP_BBOX_1, CROP_BBOX_2]
            TRUE_IMAGES_DIR = 'data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/test/full/Landsat_02'
            OUTPUT_DIR = '/home/zhaojiazhen/workspace/STF/STF/artifacts/image/subimage/CIA'
            PRED_DIRS_DICT = {
                'starfm': '/home/zhaojiazhen/workspace/STF/STF/results/starfm/syy_setting~9/CIA/one_pair~patch_size_120~patch_stride_50~window_size_51~num_classes_20/full/imgs/CIA/save_img',
                'FSDAF': '/home/zhaojiazhen/workspace/STF/STF/results/FSDAF/CIA/full',
                'FitFC': 'results/FitFC/CIA/full/FitFC_Results_L2',
                'stfdcnn': '/home/zhaojiazhen/workspace/STF/STF/results/stfdcnn/syy_setting~9/CIA/inference~stage_1_SGD_1e-2~stage_2_SGD_1e-1/full/imgs/stage_2/CIA/save_img',
                'ganstfm': '/home/zhaojiazhen/workspace/STF/STF/results/ganstfm/syy_setting-9/CIA/inference~Adam_1e-4/full/imgs/CIA/save_img',
                'stfgan': 'results/stfgan/syy_setting-9/CIA/inference~stage_1~RMSProp~stage_2~RMSProp/full/imgs/stage_2/CIA/save_img',
                'opgan': '/home/zhaojiazhen/workspace/STF/STF/results/opgan/syy_setting-9/CIA/inference~RMSProp/full/imgs/CIA/save_img',
                'swinstf': '/home/zhaojiazhen/workspace/STF/STF/results/swinstf/syy_setting-9/CIA/inference~Adam_1e-4~StepLR_step_size-15_gamma_5e-1/full/imgs/CIA/save_img',
                'fsdformer': '/home/zhaojiazhen/workspace/STF/STF/results/fsdformer/syy_setting-9/CIA/inferencer/full/imgs/CIA/save_img',
                'stfmamba': '/home/zhaojiazhen/workspace/STF/STF/results/stfmamba/syy_setting-9/CIA/inferencer/full/imgs/CIA/save_img',
                'stfdiff': 'results/stfdiff/syy_setting-9/model6_GN_SiLU/CIA/inference~Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0/full/imgs/CIA/0/save_img',
                'GPSTFDiff': '/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/lap/syy_setting-9/CIA/inference/inferency_8/full/imgs/CIA/0/save_img',
            }
        elif args.dataset == 'LGC':
            CROP_BBOX_1 = [120, 376, 140, 396]
            CROP_BBOX_2 = [700, 956, 980, 1236]
            # 将多个边框组成一个列表
            CROP_BBOXES = [CROP_BBOX_1, CROP_BBOX_2]
            TRUE_IMAGES_DIR = '/home/zhaojiazhen/workspace/STF/STF/data/spatio_temporal_fusion/LGC/private_data/syy_setting-9/test/full/Landsat_02'
            OUTPUT_DIR = '/home/zhaojiazhen/workspace/STF/STF/artifacts/image/subimage/LGC'
            PRED_DIRS_DICT = {
                'starfm': '/home/zhaojiazhen/workspace/STF/STF/results/starfm/syy_setting~9/LGC/one_pair~patch_size_120~patch_stride_50~window_size_51~num_classes_20/full/imgs/LGC/save_img',
                'FSDAF': '/home/zhaojiazhen/workspace/STF/STF/results/FSDAF/LGC/full',
                'FitFC': 'results/FitFC/LGC/full/FitFC_Results_L2',
                'stfdcnn': '/home/zhaojiazhen/workspace/STF/STF/results/stfdcnn/syy_setting~9/LGC/inference~stage_1_SGD_1e-2~stage_2_SGD_1e-1/full/imgs/stage_2/LGC/save_img',
                'ganstfm': '/home/zhaojiazhen/workspace/STF/STF/results/ganstfm/syy_setting-9/LGC/inference~Adam_1e-4/full/imgs/LGC/save_img',
                'stfgan': 'results/stfgan/syy_setting-9/LGC/inference~stage_1~RMSProp~stage_2~RMSProp/full/imgs/stage_2/LGC/save_img',
                'opgan': '/home/zhaojiazhen/workspace/STF/STF/results/opgan/syy_setting-9/LGC/inference~RMSProp/full/imgs/LGC/save_img',
                'swinstf': '/home/zhaojiazhen/workspace/STF/STF/results/swinstf/syy_setting-9/LGC/inference~Adam_1e-4~StepLR_step_size-15_gamma_5e-1/full/imgs/LGC/save_img',
                'fsdformer': '/home/zhaojiazhen/workspace/STF/STF/results/fsdformer/syy_setting-9/LGC/inferencer/full/imgs/LGC/save_img',
                'stfmamba': '/home/zhaojiazhen/workspace/STF/STF/results/stfmamba/syy_setting-9/LGC/inferencer/full/imgs/LGC/save_img',
                'stfdiff': 'results/stfdiff/syy_setting-9/model6_GN_SiLU/LGC/inference~Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0/full/imgs/LGC/0/save_img',
                'GPSTFDiff': '/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/lap/syy_setting-9/LGC/inference/full/imgs/LGC/0/save_img',
            }
        elif args.dataset == 'ML':
            CROP_BBOX_1 =[676, 932, 608, 864]
            CROP_BBOX_2 = [206, 462, 1008, 1264]
            # 将多个边框组成一个列表
            CROP_BBOXES = [CROP_BBOX_1, CROP_BBOX_2]
            TRUE_IMAGES_DIR = '/home/zhaojiazhen/workspace/STF/STF/data/spatio_temporal_fusion/ML/private_data/syy_setting-9/test/full/Landsat_02'
            OUTPUT_DIR = '/home/zhaojiazhen/workspace/STF/STF/artifacts/image/subimage/ML'
            PRED_DIRS_DICT = {
                'starfm': '/home/zhaojiazhen/workspace/STF/STF/results/starfm/syy_setting~9/ML/one_pair~patch_size_120~patch_stride_50~window_size_51~num_classes_20/full/imgs/ML/save_img',
                'FSDAF': '/home/zhaojiazhen/workspace/STF/STF/results/FSDAF/ML/full',
                'FitFC': 'results/FitFC/ML/full/FitFC_Results_L2',
                'stfdcnn': '/home/zhaojiazhen/workspace/STF/STF/results/stfdcnn/syy_setting~9/ML/inference~stage_1_SGD_1e-2~stage_2_SGD_1e-1/full/imgs/stage_2/ML/save_img',
                'ganstfm': '/home/zhaojiazhen/workspace/STF/STF/results/ganstfm/syy_setting-9/ML/inference~Adam_1e-4/full/imgs/ML/save_img',
                'stfgan': 'results/stfgan/syy_setting-9/ML/inference~stage_1~RMSProp~stage_2~RMSProp/full/imgs/stage_2/ML/save_img',
                'opgan': '/home/zhaojiazhen/workspace/STF/STF/results/opgan/syy_setting-9/ML/inference~RMSProp/full/imgs/ML/save_img',
                'swinstf': '/home/zhaojiazhen/workspace/STF/STF/results/swinstf/syy_setting-9/ML/inference~Adam_1e-4~StepLR_step_size-15_gamma_5e-1/full/imgs/ML/save_img',
                'fsdformer': '/home/zhaojiazhen/workspace/STF/STF/results/fsdformer/syy_setting-9/ML/inferencer/full/imgs/ML/save_img',
                'stfmamba': '/home/zhaojiazhen/workspace/STF/STF/results/stfmamba/syy_setting-9/ML/inferencer/full/imgs/ML/save_img',
                'stfdiff': '/home/zhaojiazhen/workspace/STF/STF/results/stfdiff/syy_setting-9/model6_GN_SiLU/ML/inference/full/imgs/ML/0/save_img',
                'GPSTFDiff': '/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/lap/syy_setting-9/ML/inference/full/imgs/ML/0/save_img',
            }
        elif args.dataset == '321':
            CROP_BBOX_1 = [1320, 1576, 760, 1016]
            CROP_BBOX_2 = [820, 1076, 900, 1156]
            CROP_BBOXES = [CROP_BBOX_1, CROP_BBOX_2]

            TRUE_IMAGES_DIR = '/home/zhaojiazhen/workspace/STF/STF/data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/test/full/Landsat_02'
    
            PRED_DIRS_DICT = {
                'DDIM_step_10' : '/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/diff_step/DDIM/inference_10step/full/imgs/CIA/0/save_img',
                'DDIM_step_100' : '/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/diff_step/DDIM/inference_100step/full/imgs/CIA/0/save_img',
                'lap_step_10' : '/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/diff_step/lap/inferency_10step/full/imgs/CIA/0/save_img',
                'lap_step_100' : '/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/syy_setting-9/CIA/ablation/diff_step/lap/inferency_100step/full/imgs/CIA/0/save_img',
            }
            OUTPUT_DIR = '/home/zhaojiazhen/workspace/STF/STF/artifacts/metric_log/abl/subimage_321'

        # 最后传入 bboxes 参数 (仅非 single 模式执行)
        generate_sub_images(
            true_dir=TRUE_IMAGES_DIR, 
            pred_dirs_dict=PRED_DIRS_DICT, 
            output_dir=OUTPUT_DIR, 
            bboxes=CROP_BBOXES
        )
