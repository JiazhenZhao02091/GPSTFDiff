import argparse
from pathlib import Path

import tifffile as tiff
from torch.nn.modules.utils import _pair
from tqdm import tqdm


def split_img(src_img, img_patch_size, stride):
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
            if src_img.ndim == 2:
                patch = src_img[h_start:h_end, w_start:w_end]
            else:
                patch = src_img[h_start:h_end, w_start:w_end, :]
            yield patch, (h_start, h_end, w_start, w_end)


def split_img_via_path(src_img_path, tar_data_dir_path, img_patch_size, stride):
    src_img_path = Path(src_img_path)
    tar_data_dir_path = Path(tar_data_dir_path)
    src_img_stem, src_img_suffix = src_img_path.stem, src_img_path.suffix
    tar_data_date_dir_path = tar_data_dir_path / src_img_stem
    tar_data_date_dir_path.mkdir(parents=True, exist_ok=True)
    data_sensor_type = src_img_stem[0]
    tar_data_name_tmpl = f"{data_sensor_type}" + r"_{}_{}_{}_{}" + f"{src_img_suffix}"
    src_img = tiff.imread(src_img_path)
    for tar_img_patch, (h_start, h_end, w_start, w_end) in split_img(
        src_img, img_patch_size, stride
    ):
        tar_data_name = tar_data_name_tmpl.format(h_start, h_end, w_start, w_end)
        tar_data_path = tar_data_date_dir_path / tar_data_name
        yield tar_data_path, tar_img_patch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split CDL data like Landsat naming"
    )
    parser.add_argument("--src_dir", type=str, required=True, help="Input CDL directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--img_patch_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_data_path_list = sorted(src_dir.glob("*.tif"))
    pbar = tqdm(src_data_path_list)
    for data_index, src_data_path in enumerate(pbar):
        pbar.set_description(
            f"split CDL: {src_data_path.name} {data_index + 1}/{len(src_data_path_list)}"
        )
        for tar_data_path, tar_data in split_img_via_path(
            src_data_path,
            out_dir,
            args.img_patch_size,
            args.stride,
        ):
            tiff.imwrite(tar_data_path, tar_data)
