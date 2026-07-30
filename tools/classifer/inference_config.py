import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


CONFIG = {
    "in_channels": 6,
    "num_classes": 5, # 重新设定为 5 个有效输出
    "ignore_label": 255,
    "device": "cuda",

    "lr": 1e-4,
    "batch_size": 2,
    "epochs": 1000,
    
    "id_map": {
        1: 0,    # 玉米
        5: 1,    # 大豆
        176: 2,  # 牧场
        141: 3, 143: 3, 
        121: 4, 122: 4, 123: 4, 124: 4 
    },
    
    "rev_map": {
        0: 1,   # 还原为玉米
        1: 5,   # 还原为大豆
        2: 176, # 还原为牧场
        3: 141, # 还原为林地
        4: 122  # 还原为低密度开发用地
    },

    
    "Landsat_path": os.path.join(PROJECT_ROOT, "data/spatio_temporal_fusion/ML/Crop/Landsat"),
    "CDL_path": os.path.join(PROJECT_ROOT, "data/spatio_temporal_fusion/ML/CDL_cropped"),
    "model_path": os.path.join(PROJECT_ROOT, "crop_unet/save_checkpoint/checkpoint_epochs_1000.pth"),
    "info_name":"checkpoint_epochs_1000.txt",
    "log_dir": os.path.join(PROJECT_ROOT, "crop_unet/log"),

}
