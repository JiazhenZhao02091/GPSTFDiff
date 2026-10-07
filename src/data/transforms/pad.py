import numpy as np
from src.utils.patch import cal_padding_img_pixel_num_hw
from torch.nn.modules.utils import _pair, _quadruple


class Pad:
    def __init__(self, key_list, patch_size, patch_stride, is_drop_last=False):
        self.key_list = key_list
        self.patch_size = _pair(patch_size)
        self.patch_stride = _pair(patch_stride)
        self.is_drop_last = is_drop_last

    def transform(self, data_info: dict):
        # 需要获取任意一个 key 的 img_size 来计算 padding
        # 假设所有图大小一样，这里用第一个 key 即可
        base_img_size = None

        for key in self.key_list:
            data = data_info[key]
            img_size = data_info[f'{key}_shape'][:2]
            if base_img_size is None:
                base_img_size = img_size
            data = self.pad(data, img_size)
            data_info[key] = data
        # --- 新增 ---
        if 'mask' in data_info and data_info['mask'] is not None:
            # Mask 需要和图像一样的 padding
            # Pad 类里的 self.pad 函数依赖 img_size，这里可以用 base_img_size
            # 小心：Mask 只有 0 和 1，Pad 如果用 'reflect' 模式可能会在边缘产生奇怪的值
            # 通常 Mask Padding 用 'constant' (填 0) 比较安全，看任务需求
            # 这里如果不改 mode，还是会用 reflect
            if base_img_size:
                data_info['mask'] = self.pad(data_info['mask'], base_img_size)

        return data_info

    def pad(self, data, img_size):
        padding_img_pixel_num = cal_padding_img_pixel_num_hw(
            img_size, self.patch_size, self.patch_stride, self.is_drop_last
        )
        data = np.pad(data, (*padding_img_pixel_num, (0, 0)), mode='reflect')
        return data

    def __call__(self, data_info: dict):
        return self.transform(data_info)
