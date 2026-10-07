import tifffile
import numpy as np
import scipy.io as sio


class LoadData:
    def __init__(self, key_list):
        self.key_list = key_list

    def transform(self, data_info: dict):
        # 存储每个数据源的“无效区域”
        invalid_masks = []
       
        for key in self.key_list:
            data_path = data_info[f'{key}_path']
            data = self.load_data(data_path)
            data_info[key] = data
            data_info[f'{key}_shape'] = data.shape
            # print(f"data type {type(data)}, data dtype: {data.dtype}")
            if data.ndim == 3: # 多通道
                c, h, w = data.shape
                if c < min(h, w):
                    data = np.transpose(data, (1, 2, 0))
                mask_single_channel = np.all(data == 0, axis=-1)
            else:              # 单通道
                mask_single_channel = (data == 0)
            invalid_masks.append(mask_single_channel)

        if invalid_masks:
            # 1. 计算所有影像空值的并集 (True 代表该位置在任意一张图中是缺失的)
            union_invalid = np.logical_or.reduce(invalid_masks)
            # 2. 转换为有效掩膜 (True 代表该位置在所有图中都有效，1有效, 0无效)
            # 增加维度 (H, W, 1) 以便后续与 (H, W, C) 的图像进行计算
            valid_mask = (~union_invalid).astype(np.float32)[..., np.newaxis]
            data_info['mask'] = valid_mask
        else:
            data_info['mask'] = None
            
        return data_info

    def load_data(self, data_path: str):
        # 确保读取的是原始数值
        data = tifffile.imread(data_path)
        return data

    def __call__(self, data_info: dict):
        return self.transform(data_info)


# SPSTFM
class LoadDictionarySparistyMatrix:
    def __init__(self, key_list, np_key_list):
        self.key_list = key_list
        self.np_key_list = np_key_list

    def transform(self, data_info: dict):
        for key in self.key_list:
            data_path = data_info[f'{key}_path']
            data = self.load_data(data_path)
            data_info[''] = data
            for np_key in self.np_key_list:
                data_info[np_key] = data[np_key]
        return data_info

    def load_data(self, data_path: str):
        data = sio.loadmat(data_path)

        return data

    def __call__(self, data_info: dict):
        return self.transform(data_info)
