import numpy as np
import torch


class Format:
    def __init__(self, key_list):
        self.key_list = key_list

    def transform(self, data_info: dict):
        for key in self.key_list:
            data = data_info[key]
            data = self.to_tensor(data)
            data_info[key] = data
        if 'mask' in data_info and data_info['mask'] is not None:
            data_info['mask'] = self.to_tensor(data_info['mask'])
        return data_info

    def to_tensor(self, data):
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        if data.ndim == 3:
            data = data.transpose(2, 0, 1)    # H W C --> C H W
        data = torch.from_numpy(data).contiguous().to(torch.float32)   
        return data

    def __call__(self, data_info: dict):
        return self.transform(data_info)

    # def __repr__(self):
    #     return self.__class__.__name__ + f'(keys={self.key_list})'
