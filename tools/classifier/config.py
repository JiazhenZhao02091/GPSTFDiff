CONFIG = {
    "in_channels": 6,
    "num_classes": 5, # 重新设定为 5 个有效输出
    "ignore_label": 255,
    "device": "cuda",

    "lr": 1e-4,
    "batch_size": 2,
    "epochs": 1000,
    
    # 映射逻辑更新
    "id_map": {
        1: 0,    # 玉米
        5: 1,    # 大豆
        176: 2,  # 牧场
        # # 林地合并
        141: 3, 143: 3, 
        # 开发用地合并
        121: 4, 122: 4, 123: 4, 124: 4 
    },
    
    # 推理还原逻辑 (选择每一组中最代表性的原始代码)
    "rev_map": {
        0: 1,   # 还原为玉米
        1: 5,   # 还原为大豆
        2: 176, # 还原为牧场
        3: 141, # 还原为林地
        4: 122  # 还原为低密度开发用地
    },

    
    "Landsat_path": "data/spatio_temporal_fusion/ML/public_processing_data/format_data/crop_118_1654_125_2173/original/Landsat",
    "CDL_path": "data/spatio_temporal_fusion/ML/CDL_cropped",
    "model_path": "checkpoints/classifier/Unet_checkpoint_epochs_1000.pth",
    "info_name":"results/classifier/Unet_checkpoint_epochs_1000.txt",
    "log_dir":  "results/classifier/log",

    # 1 6723 5224
}
