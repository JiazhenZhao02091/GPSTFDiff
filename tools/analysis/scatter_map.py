import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import argparse

# ================= 全局字体与样式设置 =================
# 确保之前安装的 Times New Roman 生效
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.unicode_minus'] = False 
# ====================================================

def clean_and_sample_data(true_img, pred_img, sample_size):
    x = true_img.astype(np.float32).flatten()
    y = pred_img.astype(np.float32).flatten()
    
    if np.max(x) > 100: x = x / 10000.0
    if np.max(y) > 100: y = y / 10000.0
        
    valid_mask = (x > 0) & (y > 0) & (~np.isnan(x)) & (~np.isnan(y))
    x, y = x[valid_mask], y[valid_mask]
    
    if len(x) > sample_size:
        idx = np.random.choice(len(x), size=sample_size, replace=False)
        return x[idx], y[idx], x, y
    return x, y, x, y 

def plot_multiband_scatter(ax, true_red_full, pred_red_full, true_nir_full, pred_nir_full, method_name, is_first_col, sample_size=30000):
    """画双波段纯色散点图"""
    tr, pr, tr_all, pr_all = clean_and_sample_data(true_red_full, pred_red_full, sample_size)
    tn, pn, tn_all, pn_all = clean_and_sample_data(true_nir_full, pred_nir_full, sample_size)

    ax.set_xlim([0, 1.0]); ax.set_ylim([0, 1.0]); ax.set_aspect('equal', 'box')
    ax.scatter(pn, tn, c='blue', s=2, alpha=0.5, edgecolors='none', label='NIR', zorder=1)
    ax.scatter(pr, tr, c='red', s=2, alpha=0.5, edgecolors='none', label='RED', zorder=2)
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, zorder=3)

    for (t_a, p_a, c, lab) in [(tr_all, pr_all, 'red', 'RED'), (tn_all, pn_all, 'blue', 'NIR')]:
        if len(t_a) > 1 and np.var(p_a) > 1e-8:
            m, b = np.polyfit(p_a.astype(np.float64), t_a.astype(np.float64), 1)
            sign = '+' if b >= 0 else '-'
            ax.plot([0, 1], [b, m*1+b], color=c, lw=2, zorder=4, label=f'{lab}: $y={m:.2f}x{sign}{abs(b):.2f}$')

    if is_first_col: ax.set_ylabel('Ground Truth', fontsize=16)
    ax.set_xlabel(method_name, fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='upper left', frameon=False, fontsize=8, handletextpad=0.1, markerscale=8)

def plot_density_scatter(ax, true_full, pred_full, band_name, method_name, is_first_col, sample_size=30000):
    """极速直方图密度散点图"""
    x_s, y_s, x_a, y_a = clean_and_sample_data(true_full, pred_full, sample_size)
    
    bins = 150
    hh, locx, locy = np.histogram2d(y_s, x_s, bins=[bins, bins], range=[[0, 1], [0, 1]])
    ix = np.clip(np.digitize(y_s, locx) - 1, 0, bins - 1)
    iy = np.clip(np.digitize(x_s, locy) - 1, 0, bins - 1)
    z = hh[ix, iy]
    
    idx = z.argsort()
    x_s, y_s, z = x_s[idx], y_s[idx], z[idx]
    
    ax.set_xlim([0, 1.0]); ax.set_ylim([0, 1.0]); ax.set_aspect('equal', 'box')
    sc = ax.scatter(y_s, x_s, c=z, s=2, cmap='jet', edgecolors='none', zorder=1)
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, zorder=2)
    
    if len(x_a) > 1 and np.var(y_a) > 1e-8:
        m, b = np.polyfit(y_a.astype(np.float64), x_a.astype(np.float64), 1)
        ax.plot([0, 1], [b, m+b], 'r-', lw=1.8, zorder=3)
        ax.text(0.05, 0.92, f'$y={m:.2f}x{"+" if b>=0 else "-"}{abs(b):.2f}$', 
                transform=ax.transAxes, fontsize=13, zorder=4)

    if is_first_col: ax.set_ylabel(band_name, fontsize=16)
    ax.set_xlabel(method_name, fontsize=14)
    ax.tick_params(axis='both', labelsize=11)
    return sc

def get_file_prefix(filename):
    basename = os.path.basename(filename)
    parts = basename.split('_')
    if len(parts) >= 2: return f"{parts[0]}_{parts[1]}"
    return basename.split('.')[0]

def extract_band(img_data, band_idx):
    if img_data.shape[0] == 6: return img_data[band_idx, :, :]
    elif img_data.shape[-1] == 6: return img_data[:, :, band_idx]
    else: raise ValueError(f"无法识别的数据维度: {img_data.shape}")

def main(dataset):
    # 路径配置
    if dataset == 'CIA':
        true_dir = 'data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/test/full/Landsat_02'
        out_dir = 'PPT/scatter_map/CIA'
        pred_dirs_dict = {
            'starfm': 'results/starfm/syy_setting~9/CIA/one_pair~patch_size_120~patch_stride_50~window_size_51~num_classes_20/full/imgs/CIA/save_img',
            'FSDAF': 'results/FSDAF/CIA/full',
            'FitFC': 'results/FitFC/CIA/full/FitFC_Results_L2',
            'stfdcnn': 'results/stfdcnn/syy_setting~9/CIA/inference~stage_1_SGD_1e-2~stage_2_SGD_1e-1/full/imgs/stage_2/CIA/save_img',
            'ganstfm': 'results/ganstfm/syy_setting-9/CIA/inference~Adam_1e-4/full/imgs/CIA/save_img',
            'stfgan': 'results/stfgan/syy_setting-9/CIA/inference~stage_1~RMSProp~stage_2~RMSProp/full/imgs/stage_2/CIA/save_img',
            'opgan': 'results/opgan/syy_setting-9/CIA/inference~RMSProp/full/imgs/CIA/save_img',
            'swinstf': 'results/swinstf/syy_setting-9/CIA/inference~Adam_1e-4~StepLR_step_size-15_gamma_5e-1/full/imgs/CIA/save_img',
            'fsdformer': 'results/fsdformer/syy_setting-9/CIA/inferencer/full/imgs/CIA/save_img',
            'stfmamba': 'results/stfmamba/syy_setting-9/CIA/inferencer/full/imgs/CIA/save_img',
            'stfdiff': 'results/stfdiff/syy_setting-9/model6_GN_SiLU/CIA/inference~Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0/full/imgs/CIA/0/save_img',
            'LapSTFDiff': 'results/LapSTFDiff/lap/syy_setting-9/CIA/inference/inferency_8/full/imgs/CIA/0/save_img',
        }
    elif dataset == 'LGC':
        true_dir = 'data/spatio_temporal_fusion/LGC/private_data/syy_setting-9/test/full/Landsat_02'
        out_dir = 'PPT/scatter_map/LGC'
        pred_dirs_dict = {
            'starfm': 'results/starfm/syy_setting~9/LGC/one_pair~patch_size_120~patch_stride_50~window_size_51~num_classes_20/full/imgs/LGC/save_img',
            'FSDAF': 'results/FSDAF/LGC/full',
            'FitFC': 'results/FitFC/LGC/full/FitFC_Results_L2',
            'stfdcnn': 'results/stfdcnn/syy_setting~9/LGC/inference~stage_1_SGD_1e-2~stage_2_SGD_1e-1/full/imgs/stage_2/LGC/save_img',
            'ganstfm': 'results/ganstfm/syy_setting-9/LGC/inference~Adam_1e-4/full/imgs/LGC/save_img',
            'stfgan': 'results/stfgan/syy_setting-9/LGC/inference~stage_1~RMSProp~stage_2~RMSProp/full/imgs/stage_2/LGC/save_img',
            'opgan': 'results/opgan/syy_setting-9/LGC/inference~RMSProp/full/imgs/LGC/save_img',
            'swinstf': 'results/swinstf/syy_setting-9/LGC/inference~Adam_1e-4~StepLR_step_size-15_gamma_5e-1/full/imgs/LGC/save_img',
            'fsdformer': 'results/fsdformer/syy_setting-9/LGC/inferencer/full/imgs/LGC/save_img',
            'stfmamba': 'results/stfmamba/syy_setting-9/LGC/inferencer/full/imgs/LGC/save_img',
            'stfdiff': 'results/stfdiff/syy_setting-9/model6_GN_SiLU/LGC/inference~Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0/full/imgs/LGC/0/save_img',
            'LapSTFDiff': 'results/LapSTFDiff/lap/syy_setting-9/LGC/inference/full/imgs/LGC/0/save_img',
        }
    else: # ML ...
        true_dir = 'data/spatio_temporal_fusion/ML/private_data/syy_setting-9/test/full/Landsat_02'
        out_dir = 'PPT/scatter_map/ML'
        pred_dirs_dict = {
            'starfm': 'results/starfm/syy_setting~9/ML/one_pair~patch_size_120~patch_stride_50~window_size_51~num_classes_20/full/imgs/ML/save_img',
            'FSDAF': 'results/FSDAF/ML/full',
            'FitFC': 'results/FitFC/ML/full/FitFC_Results_L2',
            'stfdcnn': 'results/stfdcnn/syy_setting~9/ML/inference~stage_1_SGD_1e-2~stage_2_SGD_1e-1/full/imgs/stage_2/ML/save_img',
            'ganstfm': 'results/ganstfm/syy_setting-9/ML/inference~Adam_1e-4/full/imgs/ML/save_img',
            'stfgan': 'results/stfgan/syy_setting-9/ML/inference~stage_1~RMSProp~stage_2~RMSProp/full/imgs/stage_2/ML/save_img',
            'opgan': 'results/opgan/syy_setting-9/ML/inference~RMSProp/full/imgs/ML/save_img',
            'swinstf': 'results/swinstf/syy_setting-9/ML/inference~Adam_1e-4~StepLR_step_size-15_gamma_5e-1/full/imgs/ML/save_img',
            'fsdformer': 'results/fsdformer/syy_setting-9/ML/inferencer/full/imgs/ML/save_img',
            'stfmamba': 'results/stfmamba/syy_setting-9/ML/inferencer/full/imgs/ML/save_img',
            'stfdiff': 'results/stfdiff/syy_setting-9/model6_GN_SiLU/ML/inference/full/imgs/ML/0/save_img',
            'LapSTFDiff': 'results/LapSTFDiff/lap/syy_setting-9/ML/inference/full/imgs/ML/0/save_img',
        }

    os.makedirs(out_dir, exist_ok=True)
    true_files = glob.glob(os.path.join(true_dir, '*.tif'))
    methods = list(pred_dirs_dict.keys())
    save_mode = 'both' 
    n_cols = 6 

    for true_file in true_files:
        prefix = get_file_prefix(true_file)
        print(f"Processing group: {prefix}...")
        true_data = tiff.imread(true_file)
        tr_f = extract_band(true_data, 2)
        tn_f = extract_band(true_data, 3)

        n_rows = (len(methods) + n_cols - 1) // n_cols
        
        # 1. 初始化组合图画布
        if save_mode in ['combined', 'both']:
            fig_c, axes_c = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
            axes_c_f = axes_c.flatten()
            fig_d, axes_d = plt.subplots(n_rows*2, n_cols, figsize=(4*n_cols, 4*n_rows*2))
            axes_d_f = axes_d.reshape(-1, n_cols)
            sc_ref = None

        # 2. 核心绘图循环
        for j, method in enumerate(methods):
            method_dir = pred_dirs_dict[method]
            pred_search = glob.glob(os.path.join(method_dir, f"{prefix}*.tif"))
            if not pred_search: continue
            
            pred_data = tiff.imread(pred_search[0])
            pr_f = extract_band(pred_data, 2)
            pn_f = extract_band(pred_data, 3)

            # --- Combined Mode ---
            if save_mode in ['combined', 'both']:
                plot_multiband_scatter(axes_c_f[j], tr_f, pr_f, tn_f, pn_f, method, (j%n_cols==0))
                r, c = (j//n_cols)*2, j%n_cols
                sc_ref = plot_density_scatter(axes_d_f[r, c], tr_f, pr_f, "RED", method, c==0)
                plot_density_scatter(axes_d_f[r+1, c], tn_f, pn_f, "NIR", method, c==0)

            # --- Individual Mode (随画随关，不留隐患) ---
            if save_mode in ['individual', 'both']:
                fi, ai = plt.subplots(figsize=(5,5))
                plot_multiband_scatter(ai, tr_f, pr_f, tn_f, pn_f, method, True)
                fi.savefig(os.path.join(out_dir, f'{prefix}_{method}_multi_ind.png'), dpi=300, bbox_inches='tight')
                plt.close(fi)

                fd, ad = plt.subplots(2, 1, figsize=(5,10))
                plot_density_scatter(ad[0], tr_f, pr_f, "RED", method, True)
                plot_density_scatter(ad[1], tn_f, pn_f, "NIR", method, True)
                fd.savefig(os.path.join(out_dir, f'{prefix}_{method}_dens_ind.png'), dpi=300, bbox_inches='tight')
                plt.close(fd)

        # 3. 扫尾组合图
        if save_mode in ['combined', 'both']:
            # 隐藏多余格子
            for k in range(len(methods), n_rows*n_cols):
                axes_c_f[k].axis('off')
                axes_d_f[(k//n_cols)*2, k%n_cols].axis('off')
                axes_d_f[(k//n_cols)*2+1, k%n_cols].axis('off')
            
            fig_c.tight_layout()
            fig_c.savefig(os.path.join(out_dir, f'{prefix}_multi_comb.png'), dpi=300)
            plt.close(fig_c)
            
            fig_d.tight_layout()
            fig_d.savefig(os.path.join(out_dir, f'{prefix}_dens_comb.png'), dpi=300)
            plt.close(fig_d)

    # 4. Colorbar (全局只生成一次)
    if 'sc_ref' in locals() and sc_ref:
        fc, ac = plt.subplots(figsize=(0.5, 4))
        plt.colorbar(plt.cm.ScalarMappable(norm=sc_ref.norm, cmap=sc_ref.cmap), cax=ac)
        fc.savefig(os.path.join(out_dir, 'colorbar.png'), bbox_inches='tight')
        plt.close(fc)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='LGC')
    args = argparser.parse_args()
    main(args.dataset)