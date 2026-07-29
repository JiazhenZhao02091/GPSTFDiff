"""
description: 
    CHW -> HWC
return {*}
"""
import numpy as np


def format_data(data, band):
    print(f"data.dtype: {data.dtype}")
    # 如果传入的就是float类型
    if np.issubdtype(data.dtype, np.floating):
        data = np.nan_to_num(data, nan=0.0)
        # 截断异常值，确保在 [0, 1] 范围内
        data[data > 1.0] = 1.0
        data[data < 0.0] = 0.0
        data = data.astype(np.float32)
    elif data.dtype == np.uint16:
        # 兼容旧的原始数据逻辑
        data[data > 10000] = 10000
        data[data < 0] = 0

    shape = data.shape
    if shape[0] == band:  # CHW --> HWC
        data = np.transpose(data, (1, 2, 0))
    return data


if __name__ == '__main__':
    pass
