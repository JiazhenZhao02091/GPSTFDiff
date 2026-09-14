import numpy as np
import rasterio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score, cohen_kappa_score
from scipy.stats import mode
import tifffile as tiff

def evaluate_fusion_with_viz(fused_path, cdl_path, k=12, output_img='comparison.png'):
    # 1. 读取数据
    # 使用 tifffile 读取 fused 与 CDL（兼容多种 tiff 布局）
    fused_raw = tiff.imread(fused_path)
    # 目标是得到 (C, H, W)
    if fused_raw.ndim == 2:
        fused_img = fused_raw[np.newaxis, :, :]
    elif fused_raw.ndim == 3:
        # 如果最后一维很小（通常为通道数），则视为 (H, W, C)，需要搬轴
        if fused_raw.shape[2] <= min(fused_raw.shape[0], fused_raw.shape[1]):
            fused_img = np.moveaxis(fused_raw, -1, 0)  # (C, H, W)
        else:
            # 否则认为已经是 (C, H, W) 或 (pages, rows, cols)
            fused_img = fused_raw
    else:
        raise ValueError(f"Unsupported fused image shape: {fused_raw.shape}")

    fused_img = fused_img.astype(np.float32)
    print(f"Fused image shape (C, H, W): {fused_img.shape}")

    cdl_raw = tiff.imread(cdl_path)
    # 取第一波段作为 CDL（若为多波段）
    if cdl_raw.ndim == 3:
        cdl_img = cdl_raw[0]
    else:
        cdl_img = cdl_raw
    cdl_img = np.asarray(cdl_img)
    print(f"CDL image shape: {cdl_img.shape}")

    c, h, w = fused_img.shape

    # 2. 预处理 CDL：提取前两大类，其余归 0
    # 排除背景或无效值（通常 CDL 0 为背景）
    cdl_data_flat = cdl_img.flatten()
    values, counts = np.unique(cdl_data_flat[cdl_data_flat > 0], return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    top_2_crops = values[sorted_indices[:2]]
    
    print(f"检测到的前两大类 CDL 标签为: {top_2_crops}")
    
    simplified_cdl = np.zeros_like(cdl_img)
    simplified_cdl[cdl_img == top_2_crops[0]] = 1  # 作物 A
    simplified_cdl[cdl_img == top_2_crops[1]] = 2  # 作物 B

    # 3. 掩膜处理：识别融合影像中非 0 的有效像素
    # 假设如果所有波段都是 0，则该像素为无效点
    valid_mask = np.any(fused_img != 0, axis=0)  # (h,w) 布尔型

    # 4. KMeans 聚类（有效像素）
    data_reshaped = fused_img.reshape(c, h * w).T  # (h*w, c)
    valid_flat_idx = np.flatnonzero(valid_mask)    # 1D索引
    valid_data = data_reshaped[valid_flat_idx]     # (有效像素数, c)

    print(f"正在对 {len(valid_data)} 个有效像素进行 K-Means 聚类 (k={k})...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    valid_clusters = kmeans.fit_predict(valid_data)  # (有效像素数,)

    # 5. 将聚类结果还原到原图位置
    clusters_full = np.zeros(h * w, dtype=int)
    clusters_full[valid_flat_idx] = valid_clusters + 1  # +1区分背景
    clusters_2d = clusters_full.reshape(h, w)           # (h,w)

    # 6. 类别自动映射 (Majority Voting)
    final_pred = np.zeros((h, w), dtype=clusters_2d.dtype)
    
    # 【修改】：既然 Bounds 一致，直接使用原始尺寸，移除裁剪逻辑
    simplified_crop = simplified_cdl
    clusters_crop = clusters_2d
    final_pred_crop = final_pred 
    
    print("正在进行类别映射...")
    # 逐聚类做映射
    for i in range(1, k + 1):  # 分别处理每个聚类
        mask = (clusters_crop == i)
        # mask 应与 simplified_crop shape 一致
        corresponding_cdl_values = simplified_crop[mask]
        
        if corresponding_cdl_values.size > 0:
            # 【修改】使用更直观的 numpy 统计进行多数投票
            vals, counts = np.unique(corresponding_cdl_values, return_counts=True)
            max_idx = np.argmax(counts)
            best_match = vals[max_idx]
            final_pred_crop[mask] = best_match
            
            # 调试：查看每个聚类主要映射到了什么
            # print(f"Cluster {i} -> Class {best_match} (分布: {dict(zip(vals, counts))})")

    # 7. 精度评估
    eval_mask = valid_mask
    
    # 评估 1: 全局评估 (包含背景 0)
    # 仅在 valid_mask 范围内评估
    eval_mask_flat = valid_mask.flatten()
    y_true_all = simplified_crop.flatten()[eval_mask_flat]
    y_pred_all = final_pred_crop.flatten()[eval_mask_flat]
    
    # 评估 2: 仅作物区域评估 (重点！排除背景 0)
    # 我们只关心在是田块的地方，模型有没有分对
    crop_mask_flat = (y_true_all > 0) 
    y_true_crops = y_true_all[crop_mask_flat]
    y_pred_crops = y_pred_all[crop_mask_flat]

    print("--- 评估报告 ---")
    print(">> 全局评估 (包含背景):")
    if len(y_true_all) > 0:
        oa = accuracy_score(y_true_all, y_pred_all)
        kappa = cohen_kappa_score(y_true_all, y_pred_all)
        print(f"Overall Accuracy (OA): {oa:.4f}")
        print(f"Kappa Coefficient: {kappa:.4f}")
        print(f"预测值分布: {np.unique(y_pred_all, return_counts=True)}")
    
    print(">> 仅作物区域评估 (排除背景，只看 Corn vs Soybean):")
    if len(y_true_crops) > 0:
        oa_crop = accuracy_score(y_true_crops, y_pred_crops)
        kappa_crop = cohen_kappa_score(y_true_crops, y_pred_crops)
        print(f"Crop Accuracy (OA): {oa_crop:.4f}")
        print(f"Crop Kappa: {kappa_crop:.4f}")
        print(f"作物区预测分布: {np.unique(y_pred_crops, return_counts=True)}")
    else:
        print("未检测到有效的作物像素重叠区域。")

    # 8. 可视化
    fig, axes = plt.subplots(1, 4, figsize=(24, 6)) # 增加一个聚类原始图
    rgb_viz = np.moveaxis(fused_img[:3, :, :], 0, -1)
    rgb_viz = np.clip(rgb_viz / np.percentile(rgb_viz[rgb_viz>0], 98), 0, 1)
    
    axes[0].imshow(rgb_viz)
    axes[0].set_title("Fused Image (False Color)")
    axes[0].axis('off')

    axes[1].imshow(simplified_crop, cmap='viridis', vmin=0, vmax=2)
    axes[1].set_title("Target CDL (True Labels)")
    axes[1].axis('off')
    
    # 新增：原始聚类结果（未映射）
    axes[2].imshow(clusters_crop, cmap='tab20')
    axes[2].set_title(f"Raw Clusters (k={k})")
    axes[2].axis('off')

    im2 = axes[3].imshow(final_pred_crop, cmap='viridis', vmin=0, vmax=2)
    axes[3].set_title("Mapped Result")
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f"对比图已保存至: {output_img}")
    
    return final_pred

# 使用方法:

# CDL_path = "/home/zhaojiazhen/workspace/STF/STF/data/spatio_temporal_fusion/McLeanv2/CDL/CDL_McLean_2021.tif"
# CDL_path = "/home/zhaojiazhen/workspace/STF/STF/debug/CDL_McLean_2021_crop.tif"
# PRED_path = "/home/zhaojiazhen/workspace/STF/STF/results/stfdiff/syy_setting-9/model6_GN_SiLU/McLeanv3/inference~Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0/full/imgs/McLeanv3/0/save_img/Group_02_save_img_.tif"
# PRED_path_landsat = "/home/zhaojiazhen/workspace/STF/STF/results/stfdiff/syy_setting-9/model6_GN_SiLU/McLeanv3/inference~Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0/full/imgs/McLeanv3/0/save_img/Group_02_save_img_.tif"
# PRED_path_modis = "/home/zhaojiazhen/workspace/STF/STF/data/spatio_temporal_fusion/McLeanv3/raw_data/MODIS/MODIS_McLean_2021_05_05.tif"
# PRED_path_landsat = "/home/zhaojiazhen/workspace/STF/STF/data/spatio_temporal_fusion/McLeanv3/raw_data/Landsat/Landsat_8_McLean_2021_05_05.tif"
# PRED_path_landsat = "/home/zhaojiazhen/workspace/STF/STF/data/spatio_temporal_fusion/McLeanv3/raw_data/Landsat/Landsat_8_McLean_2020_08_06.tif"
# PRED_path_landsat = "/home/zhaojiazhen/workspace/STF/STF/data/spatio_temporal_fusion/tmp_Landsat/output/Landsat_8_McLean_2021_05_05.tif"
# # print("####################################################")
# # print(f"-- Evaluating With True Image: MODIS and Landsat --")
# # print("####################################################")
# # print("----------------------------------------------------")
# # print("MODIS Evaluation:")
# # print(f"MODIS Shape is {tiff.imread(PRED_path_modis).shape}")
# # res_modis = evaluate_fusion_with_viz(PRED_path_modis, CDL_path, k=8)
# # print("----------------------------------------------------")
# # print("Landsat Evaluation:")
# # print(f"Landsat Shape is {tiff.imread(PRED_path_landsat).shape}")
# # res_landsat = evaluate_fusion_with_viz(PRED_path_landsat, CDL_path, k=8)



# res_landsat = evaluate_fusion_with_viz(PRED_path, CDL_path, k=8)
# with rasterio.open(PRED_path) as src_f, rasterio.open(CDL_path) as src_c:
#     print(f"Fused Bounds: {src_f.bounds}")
#     print(f"CDL Bounds:   {src_c.bounds}")
#     print(f"Fused Transform: {src_f.transform}")
#     print(f"CDL Transform:   {src_c.transform}")

if __name__ == "__main__":
    pred_img = "/home/zhaojiazhen/workspace/STF/STF/results/stfdiff/syy_setting-9/model6_GN_SiLU/McLeanv3/inference~Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0/full/imgs/McLeanv3/0/save_img/Group_02_save_img_.tif"
    modis_img = "/home/zhaojiazhen/workspace/STF/STF/debug/MODIS_McLean_2021_05_05_crop.tif"
    landsat_img = "/home/zhaojiazhen/workspace/STF/STF/debug/Landsat_8_McLean_2021_05_05_crop.tif"
    cdl_path = "/home/zhaojiazhen/workspace/STF/STF/debug/CDL_McLean_2021_crop.tif"
    evaluate_fusion_with_viz(pred_img, cdl_path, k=8, output_img='pred_comparison.png')
    evaluate_fusion_with_viz(modis_img, cdl_path, k=8, output_img='modis_comparison.png')
    evaluate_fusion_with_viz(landsat_img, cdl_path, k=8, output_img='landsat_comparison.png')