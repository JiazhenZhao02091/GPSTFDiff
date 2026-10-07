import numpy as np


def format_data(data, band):
    print(f"data.dtype: {data.dtype}")

    if np.issubdtype(data.dtype, np.floating):
        data = np.nan_to_num(data, nan=0.0)

        data[data > 1.0] = 1.0
        data[data < 0.0] = 0.0
        data = data.astype(np.float32)
    elif data.dtype == np.uint16:

        data[data > 10000] = 10000
        data[data < 0] = 0

    shape = data.shape
    if shape[0] == band:
        data = np.transpose(data, (1, 2, 0))
    return data


if __name__ == '__main__':
    pass
