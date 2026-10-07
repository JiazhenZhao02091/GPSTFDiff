import cv2
import numpy as np


class Flip:
    def __init__(self, key_list):
        self.key_list = key_list

    def transform(self, data_info: dict):
        flip_mode = np.random.randint(0, 2)
        is_flip = np.random.randint(0, 2)
        for key in self.key_list:
            data = data_info[key]
            data = self.flip(data, is_flip, flip_mode)
            data_info[key] = data
            
        # 同步处理 Mask (如果存在)
        if 'mask' in data_info and data_info['mask'] is not None:
             data_info['mask'] = self.flip(data_info['mask'], is_flip, flip_mode)
             # Mask 在 LoadData 生成时是 (H, W, 1)，cv2.flip 可能返回 (H, W)，需注意保持维度
             if data_info['mask'].ndim == 2:
                 data_info['mask'] = data_info['mask'][..., np.newaxis]

        data_info['flip_mode'] = flip_mode if is_flip else -1
        return data_info

    def flip(self, data, is_flip, flip_mode):
        if is_flip:
            data = cv2.flip(data, flip_mode)
        return data

    def __call__(self, data_info: dict):
        return self.transform(data_info)

    # def __repr__(self):
    #     return self.__class__.__name__ + f'(keys={self.key_list})'
