# DATASET_TYPE = ['AHB', 'Daxing', 'Tianjin', 'CIA', 'LGC', 'ML']
DATASET_TYPE = ['ML', 'CIA', 'LGC']
# DATASET_TYPE = ['CIA', 'LGC']
# DATASET_TYPE = ['AHB', 'Daxing', 'Tianjin']
# DATASET_TYPE = ['McLeanv3', 'McLean']
# DATASET_TYPE = ['McLeanv3']
# DATASET_TYPE = ['McLeanv4']
# DATASET_TYPE = ['ML']

SENSOR_TYPE = ['Landsat', 'MODIS']
BAND = {'AHB': 6, 'Daxing': 6, 'Tianjin': 6, 'CIA': 6, 'LGC': 6, 'McLean': 6, 'McLeanv2': 6, 'McLeanv4': 6, 'McLeanv5': 6, 'ML': 6}

CROP_INFO = {
    'AHB': {'is_crop': False, 'crop_shift': [0, 0], 'crop_size': [0, 0]},
    'Daxing': {'is_crop': False, 'crop_shift': [0, 0], 'crop_size': [0, 0]},
    'Tianjin': {'is_crop': False, 'crop_shift': [0, 0], 'crop_size': [0, 0]},
    'CIA': {'is_crop': True, 'crop_shift': [20, 106], 'crop_size': [1792, 1280]},
    'LGC': {'is_crop': True, 'crop_shift': [40, 32], 'crop_size': [2560, 3072]},
    'McLean': {'is_crop': True, 'crop_shift': [50, 50], 'crop_size': [1536, 2048]},
    'McLeanv2': {'is_crop': True, 'crop_shift': [50, 50], 'crop_size': [1536, 2048]},
    'McLeanv3': {'is_crop': True, 'crop_shift': [50, 50], 'crop_size': [1536, 2048]},
    'McLeanv4': {'is_crop': True, 'crop_shift': [50, 50], 'crop_size': [1536, 2048]},
    'McLeanv5': {'is_crop': True, 'crop_shift': [50, 50], 'crop_size': [1536, 2048]},
    'ML': {'is_crop': True, 'crop_shift': [118, 125], 'crop_size': [1536, 2048]},
}

# 209: McLean未优化的版本;   v2：0.01 zero_max_ratio; v3: Optim MODIS 0.01 == tzg v2
# tzg: McLean优化后的MODIS； v2: 0.01 zero_max_ratio; v3: mask计算loss 
