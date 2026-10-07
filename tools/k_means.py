def evaluate_fusion_with_viz(fused_path, cdl_path, k=12, output_img='comparison.png'):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.metrics import accuracy_score, cohen_kappa_score
    import tifffile as tiff


    fused_raw = tiff.imread(fused_path)

    if fused_raw.ndim == 2:
        fused_img = fused_raw[np.newaxis, :, :]
    elif fused_raw.ndim == 3:

        if fused_raw.shape[2] <= min(fused_raw.shape[0], fused_raw.shape[1]):
            fused_img = np.moveaxis(fused_raw, -1, 0)
        else:

            fused_img = fused_raw
    else:
        raise ValueError(f"Unsupported fused image shape: {fused_raw.shape}")

    fused_img = fused_img.astype(np.float32)
    print(f"Fused image shape (C, H, W): {fused_img.shape}")

    cdl_raw = tiff.imread(cdl_path)

    if cdl_raw.ndim == 3:
        cdl_img = cdl_raw[0]
    else:
        cdl_img = cdl_raw
    cdl_img = np.asarray(cdl_img)
    print(f"CDL image shape: {cdl_img.shape}")

    c, h, w = fused_img.shape


    cdl_data_flat = cdl_img.flatten()
    values, counts = np.unique(cdl_data_flat[cdl_data_flat > 0], return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    top_2_crops = values[sorted_indices[:2]]

    print(f"检测到的前两大类 CDL 标签为: {top_2_crops}")

    simplified_cdl = np.zeros_like(cdl_img)
    simplified_cdl[cdl_img == top_2_crops[0]] = 1
    simplified_cdl[cdl_img == top_2_crops[1]] = 2


    valid_mask = np.any(fused_img != 0, axis=0)


    data_reshaped = fused_img.reshape(c, h * w).T
    valid_flat_idx = np.flatnonzero(valid_mask)
    valid_data = data_reshaped[valid_flat_idx]

    print(f"正在对 {len(valid_data)} 个有效像素进行 K-Means 聚类 (k={k})...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    valid_clusters = kmeans.fit_predict(valid_data)


    clusters_full = np.zeros(h * w, dtype=int)
    clusters_full[valid_flat_idx] = valid_clusters + 1
    clusters_2d = clusters_full.reshape(h, w)


    final_pred = np.zeros((h, w), dtype=clusters_2d.dtype)


    simplified_crop = simplified_cdl
    clusters_crop = clusters_2d
    final_pred_crop = final_pred

    print("正在进行类别映射...")

    for i in range(1, k + 1):
        mask = (clusters_crop == i)

        corresponding_cdl_values = simplified_crop[mask]

        if corresponding_cdl_values.size > 0:

            vals, counts = np.unique(corresponding_cdl_values, return_counts=True)
            max_idx = np.argmax(counts)
            best_match = vals[max_idx]
            final_pred_crop[mask] = best_match


    eval_mask = valid_mask


    eval_mask_flat = valid_mask.flatten()
    y_true_all = simplified_crop.flatten()[eval_mask_flat]
    y_pred_all = final_pred_crop.flatten()[eval_mask_flat]


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


    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    rgb_viz = np.moveaxis(fused_img[:3, :, :], 0, -1)
    rgb_viz = np.clip(rgb_viz / np.percentile(rgb_viz[rgb_viz>0], 98), 0, 1)

    axes[0].imshow(rgb_viz)
    axes[0].set_title("Fused Image (False Color)")
    axes[0].axis('off')

    axes[1].imshow(simplified_crop, cmap='viridis', vmin=0, vmax=2)
    axes[1].set_title("Target CDL (True Labels)")
    axes[1].axis('off')


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


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Evaluate fusion images against CDL labels with K-Means')
    parser.add_argument('--pred-image', required=True)
    parser.add_argument('--modis-image')
    parser.add_argument('--landsat-image')
    parser.add_argument('--cdl', required=True)
    parser.add_argument('--clusters', type=int, default=8)
    parser.add_argument('--output-dir', default='results/classification')
    args = parser.parse_args()
    if args.clusters < 2:
        parser.error('clusters must be at least 2')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, image in [('pred', args.pred_image), ('modis', args.modis_image), ('landsat', args.landsat_image)]:
        if image:
            evaluate_fusion_with_viz(image, args.cdl, k=args.clusters,
                                     output_img=str(output / f'{name}_comparison.png'))
