from src.utils.img.process.linear_stretch import truncated_linear_stretch
import argparse
import tifffile as tiff
from pathlib import Path
from scripts.spatio_temparol_fusion.constant import *
from tqdm import tqdm
from typing import Any
from torch.nn.modules.utils import _pair
import random 
import numpy as np # 用于计算像素占比

# HWC
def split_img(
    src_img,
    img_patch_size,
    stride,
    max_zero_ratio=1.0, # 默认1.0表示不进行过滤
):
    img_patch_size = _pair(img_patch_size)
    stride = _pair(stride)
    h, w = src_img.shape[:2]
    h_num = (
        (h - img_patch_size[0]) // stride[0] + 1
        if (h - img_patch_size[0]) % stride[0] == 0
        else (h - img_patch_size[0]) // stride[0] + 2
    )
    w_num = (
        (w - img_patch_size[1]) // stride[1] + 1
        if (w - img_patch_size[1]) % stride[1] == 0
        else (w - img_patch_size[1]) // stride[1] + 2
    )
    for h_index in range(h_num):
        for w_index in range(w_num):
            h_start = h_index * stride[0]
            w_start = w_index * stride[1]
            h_end = h_start + img_patch_size[0]
            w_end = w_start + img_patch_size[1]
            if h_end > h:
                h_start = h - img_patch_size[0]
                h_end = h
            if w_end > w:
                w_start = w - img_patch_size[1]
                w_end = w
            
            # 提取 Patch
            patch = src_img[h_start:h_end, w_start:w_end, :]

            # 计算无效像素（全0）占比
            if patch.ndim == 3:
                # 如果是多波段，要求所有波段都为0才算无效像素（根据具体需求，也可以是任意波段为0）
                # 这里假设背景区域所有波段都是0
                invalid_mask = np.all(patch == 0, axis=-1)
            else:
                invalid_mask = (patch == 0)
            
            current_zero_ratio = np.mean(invalid_mask)

            # 只有当无效像素占比小于等于阈值时才保留
            if current_zero_ratio <= max_zero_ratio:
                yield patch, (
                    h_start,
                    h_end,
                    w_start,
                    w_end,
                )


def split_img_random(src_img, img_patch_size, num_patches, max_zero_ratio=1.0):
    
    """
    随机裁切 num_patches 个图像块。
    如果 Patch 中 0 值像素占比超过 max_zero_ratio，则重新尝试采样。
    """
    img_patch_size = _pair(img_patch_size)
    h, w = src_img.shape[:2]
    patch_h, patch_w = img_patch_size
    
    # Check if image is smaller than patch size
    if h < patch_h or w < patch_w:
        raise ValueError(f"Image size ({h}, {w}) is smaller than patch size ({patch_h}, {patch_w})")

    count = 0
    max_attempts = num_patches * 50 # 最大尝试次数，防止死循环
    attempts = 0

    while count < num_patches and attempts < max_attempts:
        attempts += 1
        h_start = random.randint(0, h - patch_h)
        w_start = random.randint(0, w - patch_w)
        h_end = h_start + patch_h
        w_end = w_start + patch_w
        
        patch = src_img[h_start:h_end, w_start:w_end, :]
        
        # 计算无效像素占比
        if patch.ndim == 3:
            invalid_mask = np.all(patch == 0, axis=-1)
        else:
            invalid_mask = (patch == 0)
            
        if np.mean(invalid_mask) <= max_zero_ratio:
            count += 1
            yield patch, (
                h_start,
                h_end,
                w_start,
                w_end,
            )


def split_img_via_path(src_img_path, tar_data_dir_path, img_patch_size, stride, max_zero_ratio):
    src_img_path = Path(src_img_path)
    tar_data_dir_path = Path(tar_data_dir_path)
    src_img_stem, src_img_suffix = src_img_path.stem, src_img_path.suffix
    tar_data_date_dir_path = tar_data_dir_path / src_img_stem
    tar_data_date_dir_path.mkdir(parents=True, exist_ok=True)
    data_sensor_type = src_img_stem[0]
    tar_data_name_tmpl = f'{data_sensor_type}' + r'_{}_{}_{}_{}' + f'{src_img_suffix}'
    src_img = tiff.imread(src_img_path)
    for tar_img_patch, (h_start, h_end, w_start, w_end) in split_img(
        src_img, img_patch_size, stride, max_zero_ratio
    ):
        tar_data_name = tar_data_name_tmpl.format(h_start, h_end, w_start, w_end)
        tar_data_path = tar_data_date_dir_path / tar_data_name
        yield tar_data_path, tar_img_patch


def split_img_random_via_path(src_img_path, tar_data_dir_path, img_patch_size, num_patches, max_zero_ratio):
    src_img_path = Path(src_img_path)
    tar_data_dir_path = Path(tar_data_dir_path)
    src_img_stem, src_img_suffix = src_img_path.stem, src_img_path.suffix
    tar_data_date_dir_path = tar_data_dir_path / src_img_stem
    tar_data_date_dir_path.mkdir(parents=True, exist_ok=True)
    data_sensor_type = src_img_stem[0]
    tar_data_name_tmpl = f'{data_sensor_type}' + r'_{}_{}_{}_{}' + f'{src_img_suffix}'
    src_img = tiff.imread(src_img_path)
    
    for tar_img_patch, (h_start, h_end, w_start, w_end) in split_img_random(
        src_img, img_patch_size, num_patches, max_zero_ratio
    ):
        tar_data_name = tar_data_name_tmpl.format(h_start, h_end, w_start, w_end)
        tar_data_path = tar_data_date_dir_path / tar_data_name
        yield tar_data_path, tar_img_patch


# python -m scripts.spatio_temparol_fusion.process.split --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data --tar_data_prefix public_processing_data/format_data/split_size_{}_stride_{} --img_patch_size 256 --stride 128
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='format data')
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--src_data_prefix', type=str, required=True)
    parser.add_argument('--tar_data_prefix', type=str, required=True)
    parser.add_argument('--img_patch_size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=128)
    
    # 模式选择
    parser.add_argument('--mode', type=str, default='sliding', choices=['sliding', 'random'], help='Split mode: sliding window or random crop')
    # 随机模式下的 Patch 数量
    parser.add_argument('--num_patches', type=int, default=50, help='Number of patches for random crop mode')
    # 无效像素（0值）过滤阈值，默认 0.05 (5%)
    parser.add_argument('--max_zero_ratio', type=float, default=0.05, help='Discard patch if ratio of 0-value pixels > threshold (default 0.05)')
    
    args = parser.parse_args()

    root_path = Path(args.root_path)
    src_data_prefix_tmpl = args.src_data_prefix
    tar_data_prefix_tmpl = args.tar_data_prefix
    img_patch_size = args.img_patch_size
    stride = args.stride
    mode = args.mode
    num_patches = args.num_patches
    max_zero_ratio = args.max_zero_ratio

    print(f'split mode: {mode}, max_zero_ratio: {max_zero_ratio}')

    for dataset_type in DATASET_TYPE:
        for sensor_type in SENSOR_TYPE:
            if 'crop' in src_data_prefix_tmpl:
                crop_info = CROP_INFO[dataset_type]
                crop_shift = crop_info['crop_shift']
                crop_size = crop_info['crop_size']
                crop_top = crop_shift[0]
                crop_bottom = crop_top + crop_size[0]
                crop_left = crop_shift[1]
                crop_right = crop_left + crop_size[1]
                src_data_prefix = src_data_prefix_tmpl.format(
                    crop_top, crop_bottom, crop_left, crop_right
                )
                tar_data_prefix = tar_data_prefix_tmpl.format(
                    crop_top, crop_bottom, crop_left, crop_right, img_patch_size, stride
                )
            else:
                src_data_prefix = src_data_prefix_tmpl
                if mode == 'sliding':
                    tar_data_prefix = tar_data_prefix_tmpl.format(img_patch_size, stride)
                else:
                    # Modify folder name for random crop to distinguish it
                    tar_data_prefix = tar_data_prefix_tmpl.replace("stride_{}", "").replace("split_size_{}", "random_cnt_{}_size_{}")
                    tar_data_prefix = tar_data_prefix.format(num_patches, img_patch_size)

            src_data_dir_path = (
                root_path / dataset_type / src_data_prefix / f'original' / sensor_type
            )
            tar_data_dir_path = (
                root_path / dataset_type / tar_data_prefix / 'original' / sensor_type
            )
            tar_data_dir_path.mkdir(parents=True, exist_ok=True)
            src_data_path_list = list(src_data_dir_path.glob('*.tif'))
        
            print(f"tar: {tar_data_dir_path}, src: {src_data_dir_path}")
            pbar = tqdm(src_data_path_list)
            for data_index, src_data_path in enumerate(pbar):
                pbar.set_description(
                    f'format {dataset_type} {sensor_type}: {src_data_path.name} {data_index + 1}/{len(src_data_path_list)}'
                )
                
                if mode == 'sliding':
                    splitter = split_img_via_path(
                        src_data_path,
                        tar_data_dir_path,
                        img_patch_size,
                        stride,
                        max_zero_ratio,
                    )
                else:
                    splitter = split_img_random_via_path(
                        src_data_path,
                        tar_data_dir_path,
                        img_patch_size,
                        num_patches,
                        max_zero_ratio,
                    )

                for tar_data_path, tar_data in splitter:
                    tiff.imwrite(tar_data_path, tar_data)

    # 修复问题：1. 硬编码路径 2. 字符串类型路径无法调用 .glob 3. 缺乏目录存在性处理、可复用性
    # 假定此处已有 argparse 解析 --root_path, --src_data_prefix, --tar_data_prefix 等参数

    # # 获取路径，转为 Path，确保可用
    # src_data_dir_path = Path(args.root_path) / args.src_data_prefix
    # tar_data_dir_path = Path(
    #     args.tar_data_prefix.format(args.img_patch_size, args.stride)
    #     if '{}' in args.tar_data_prefix
    #     else args.tar_data_prefix
    # )
    # tar_data_dir_path = Path(tar_data_dir_path)
    # tar_data_dir_path.mkdir(parents=True, exist_ok=True)
    # src_data_path_list = list(src_data_dir_path.glob('*.tif'))

    # print(f"tar: {tar_data_dir_path}, src: {src_data_dir_path}")
    # pbar = tqdm(src_data_path_list)
    # for data_index, src_data_path in enumerate(pbar):
    #     pbar.set_description(
    #         f'format :{src_data_path.name} {data_index+1}/{len(src_data_path_list)}'
    #     )

    #     # 读取图像，自动扩展为 HWC 格式（即使单通道只读为HW）
    #     src_img = tiff.imread(src_data_path)
    #     if src_img.ndim == 2:
    #         src_img = src_img[:, :, None]  # 转为HWC

    #     # 采用分割
    #     for idx, (patch, (h_start, h_end, w_start, w_end)) in enumerate(
    #         split_img(
    #             src_img,
    #             args.img_patch_size,
    #             args.stride,
    #             args.max_zero_ratio if hasattr(args, "max_zero_ratio") else 1.0,
    #         )
    #     ):
    #         patch_name = (
    #             f"{src_data_path.stem}_patch_{h_start}_{h_end}_{w_start}_{w_end}.tif"
    #         )
    #         out_path = tar_data_dir_path / patch_name
    #         out_path.parent.mkdir(parents=True, exist_ok=True)
    #         tiff.imwrite(str(out_path), patch)


# python -m scripts.spatio_temparol_fusion.process.split \
#   --root_path data/spatio_temporal_fusion/McLean/raw_data \
#   --src_data_prefix CDL \
#   --tar_data_prefix data/spatio_temporal_fusion/McLean/raw_data/split_size_{}_stride_{} \
#   --img_patch_size 256 \
#   --stride 128 \