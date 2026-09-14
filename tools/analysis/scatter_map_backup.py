"""
功能:
    scatter_map 的备用版本，使用 gaussian_kde 生成 RED/NIR 密度散点图，并保存组合图和单模型图。

参数:
    --dataset (str): 数据集名称，可选 CIA、LGC、ML。
    true_dir (str): main 中按数据集配置的真实影像目录。
    pred_dirs_dict (dict[str, str]): 模型名称到预测影像目录的映射。
    band_config (dict[str, int]): RED/NIR 波段索引。
    save_mode (str): 输出模式，可保存组合图、单图或两者。

返回:
    None。main 函数不返回数据对象。

输出:
    输出 multiband、density 和 colorbar PNG 文件。
"""
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, gaussian_kde
import argparse

# ================= 新增：全局字体设置 =================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 14
# ====================================================

def clean_and_sample_data(true_img, pred_img, sample_size):
    x = true_img.astype(np.float32).flatten()
    y = pred_img.astype(np.float32).flatten()
    
    if np.max(x) > 100:  
        x = x / 10000.0
    if np.max(y) > 100:
        y = y / 10000.0
        
    valid_mask = (x > 0) & (y > 0) & (~np.isnan(x)) & (~np.isnan(y))
    x, y = x[valid_mask], y[valid_mask]
    
    if len(x) > sample_size:
        idx = np.random.choice(len(x), size=sample_size, replace=False)
        x_sample, y_sample = x[idx], y[idx]
    else:
        x_sample, y_sample = x, y
        
    return x_sample, y_sample, x, y 

def plot_multiband_scatter(ax, true_red_full, pred_red_full, true_nir_full, pred_nir_full, method_name, is_first_col, sample_size=30000):
    """画双波段纯色散点图"""
    true_red, pred_red, true_red_all, pred_red_all = clean_and_sample_data(true_red_full, pred_red_full, sample_size)
    true_nir, pred_nir, true_nir_all, pred_nir_all = clean_and_sample_data(true_nir_full, pred_nir_full, sample_size)

    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 1.0])
    ax.set_aspect('equal', 'box')

    scatter_nir = ax.scatter(pred_nir, true_nir, c='blue', s=2, alpha=0.6, edgecolors='none', label='NIR', zorder=1)
    scatter_red = ax.scatter(pred_red, true_red, c='red', s=2, alpha=0.6, edgecolors='none', label='RED', zorder=2)

    ax.plot([0, 1], [0, 1], 'k--', lw=2.0, zorder=3)

    legend_handles, legend_labels = [], []

    if len(true_red_all) > 1 and np.var(pred_red_all) > 1e-8:
        m_r, b_r = np.polyfit(pred_red_all.astype(np.float64), true_red_all.astype(np.float64), 1)
        line_red, = ax.plot([0, 1], [b_r, m_r*1 + b_r], 'r-', lw=2.5, zorder=4)
        legend_handles.append(line_red)
        legend_labels.append(f'RED: $y={m_r:.2f}x + {b_r:.2f}$')

    if len(true_nir_all) > 1 and np.var(pred_nir_all) > 1e-8:
        m_n, b_n = np.polyfit(pred_nir_all.astype(np.float64), true_nir_all.astype(np.float64), 1)
        line_nir, = ax.plot([0, 1], [b_n, m_n*1 + b_n], 'b-', lw=2.5, zorder=4)
        legend_handles.append(line_nir)
        legend_labels.append(f'NIR: $y={m_n:.2f}x + {b_n:.2f}$')

    legend_handles.extend([scatter_red, scatter_nir])
    legend_labels.extend(['RED', 'NIR'])

    if is_first_col:
        ax.set_ylabel('Ground Truth', fontsize=18)
    ax.set_xlabel(method_name, fontsize=18)
    
    ax.set_xticks(np.arange(0, 1.2, 0.2))
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    ax.tick_params(axis='both', labelsize=16)

    leg = ax.legend(legend_handles, legend_labels, loc='upper left', frameon=False, fontsize=14, handletextpad=0.2)
    for handle in leg.legend_handles:
        if isinstance(handle, plt.matplotlib.collections.PathCollection):
            handle.set_sizes([60.0])  
            handle.set_alpha(1.0)


def plot_density_scatter(ax, true_full, pred_full, band_name, method_name, is_first_col, sample_size=30000):
    """画单波段密度着色散点图"""
    # KDE 计算极慢，保持较小的降采样数值（如 30000）
    x_samp, y_samp, x_all, y_all = clean_and_sample_data(true_full, pred_full, sample_size)
    
    # 算密度
    xy = np.vstack([x_samp, y_samp])
    z = gaussian_kde(xy)(xy)
    
    # 排序散点，使密度高的画在最上面
    idxArr = z.argsort()
    x_samp, y_samp, z = x_samp[idxArr], y_samp[idxArr], z[idxArr]
    
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 1.0])
    ax.set_aspect('equal', 'box')
    
    scatter = ax.scatter(y_samp, x_samp, c=z, s=2, cmap='jet', edgecolors='none', zorder=1)
    ax.plot([0, 1], [0, 1], 'k--', lw=2.0, zorder=2)
    
    if len(x_all) > 1 and np.var(y_all) > 1e-8:
        m, b = np.polyfit(y_all.astype(np.float64), x_all.astype(np.float64), 1)
        ax.plot([0, 1], [b, m*1 + b], 'r-', lw=2.0, zorder=3)
        
        textstr = f'$y = {m:.2f}x + {b:.2f}$'
        # props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=16,
                verticalalignment='top', zorder=4)

    if is_first_col:
        ax.set_ylabel(f'{band_name}', fontsize=16)
    
    # 只有对于第二行(红波段一般放第一行，近红外放第二行，这里用统一的方法名代替)我们再加 X 标签
    ax.set_xlabel(f'{method_name}', fontsize=16)
    ax.set_xticks(np.arange(0, 1.2, 0.2))
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    ax.tick_params(axis='both', labelsize=14)
    
    return scatter


def get_file_prefix(filename):
    basename = os.path.basename(filename)
    parts = basename.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return basename.split('.')[0]


def extract_band(img_data, band_idx):
    if img_data.shape[0] == 6:
        return img_data[band_idx, :, :]
    elif img_data.shape[-1] == 6:
        return img_data[:, :, band_idx]
    else:
        raise ValueError(f"无法识别的数据维度: {img_data.shape}，请确保是 6 波段图像。")

def main(dataset):
    if dataset == 'CIA':
        true_dir = '/home/zhaojiazhen/workspace/STF/STF/data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/test/full/Landsat_02'
        out_dir = 'artifacts/image/scatter_map/CIA'
        pred_dirs_dict = {
            'starfm': '/home/zhaojiazhen/workspace/STF/STF/results/starfm/syy_setting~9/CIA/one_pair~patch_size_120~patch_stride_50~window_size_51~num_classes_20/full/imgs/CIA/save_img',
            'FSDAF': '/home/zhaojiazhen/workspace/STF/STF/results/FSDAF/CIA/full',
            'FitFC': 'results/FitFC/CIA/full/FitFC_Results_L2',
            'stfdcnn': '/home/zhaojiazhen/workspace/STF/STF/results/stfdcnn/syy_setting~9/CIA/inference~stage_1_SGD_1e-2~stage_2_SGD_1e-1/full/imgs/stage_2/CIA/save_img',
            'ganstfm': '/home/zhaojiazhen/workspace/STF/STF/results/ganstfm/syy_setting-9/CIA/inference~Adam_1e-4/full/imgs/CIA/save_img',
            'stfgan': 'results/stfgan/syy_setting-9/CIA/inference~stage_1~RMSProp~stage_2~RMSProp/full/imgs/stage_2/CIA/save_img',
            'opgan': '/home/zhaojiazhen/workspace/STF/STF/results/opgan/syy_setting-9/CIA/inference~RMSProp/full/imgs/CIA/save_img',
            'swinstf': '/home/zhaojiazhen/workspace/STF/STF/results/swinstf/syy_setting-9/CIA/inference~Adam_1e-4~StepLR_step_size-15_gamma_5e-1/full/imgs/CIA/save_img',
            'fsdformer': '/home/zhaojiazhen/workspace/STF/STF/results/fsdformer/syy_setting-9/CIA/inferencer/full/imgs/CIA/save_img',
            'stfmamba': '/home/zhaojiazhen/workspace/STF/STF/results/stfmamba/syy_setting-9/CIA/inferencer/full/imgs/CIA/save_img',
            'stfdiff': 'results/stfdiff/syy_setting-9/model6_GN_SiLU/CIA/inference~Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0/full/imgs/CIA/0/save_img',
            'GPSTFDiff': '/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/lap/syy_setting-9/CIA/inference/inferency_8/full/imgs/CIA/0/save_img',
        }
    elif dataset == 'LGC':
        true_dir = '/home/zhaojiazhen/workspace/STF/STF/data/spatio_temporal_fusion/LGC/private_data/syy_setting-9/test/full/Landsat_02'
        out_dir = 'artifacts/image/scatter_map/LGC'
        pred_dirs_dict = {
            'starfm': '/home/zhaojiazhen/workspace/STF/STF/results/starfm/syy_setting~9/LGC/one_pair~patch_size_120~patch_stride_50~window_size_51~num_classes_20/full/imgs/LGC/save_img',
            'FSDAF': '/home/zhaojiazhen/workspace/STF/STF/results/FSDAF/LGC/full',
            'FitFC': 'results/FitFC/LGC/full/FitFC_Results_L2',
            'stfdcnn': '/home/zhaojiazhen/workspace/STF/STF/results/stfdcnn/syy_setting~9/LGC/inference~stage_1_SGD_1e-2~stage_2_SGD_1e-1/full/imgs/stage_2/LGC/save_img',
            'ganstfm': '/home/zhaojiazhen/workspace/STF/STF/results/ganstfm/syy_setting-9/LGC/inference~Adam_1e-4/full/imgs/LGC/save_img',
            'stfgan': 'results/stfgan/syy_setting-9/LGC/inference~stage_1~RMSProp~stage_2~RMSProp/full/imgs/stage_2/LGC/save_img',
            'opgan': '/home/zhaojiazhen/workspace/STF/STF/results/opgan/syy_setting-9/LGC/inference~RMSProp/full/imgs/LGC/save_img',
            'swinstf': '/home/zhaojiazhen/workspace/STF/STF/results/swinstf/syy_setting-9/LGC/inference~Adam_1e-4~StepLR_step_size-15_gamma_5e-1/full/imgs/LGC/save_img',
            'fsdformer': '/home/zhaojiazhen/workspace/STF/STF/results/fsdformer/syy_setting-9/LGC/inferencer/full/imgs/LGC/save_img',
            'stfmamba': '/home/zhaojiazhen/workspace/STF/STF/results/stfmamba/syy_setting-9/LGC/inferencer/full/imgs/LGC/save_img',
            'stfdiff': 'results/stfdiff/syy_setting-9/model6_GN_SiLU/LGC/inference~Adam_1e-3~PredNoiseNet_64_depth_3~timesteps_100~ddim_50~pred_x0/full/imgs/LGC/0/save_img',
            'GPSTFDiff': '/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/lap/syy_setting-9/LGC/inference/full/imgs/LGC/0/save_img',
        }
    elif dataset == 'ML': 
        true_dir = '/home/zhaojiazhen/workspace/STF/STF/data/spatio_temporal_fusion/ML/private_data/syy_setting-9/test/full/Landsat_02'
        out_dir = 'artifacts/image/scatter_map/ML'
        pred_dirs_dict = {
            'starfm': '/home/zhaojiazhen/workspace/STF/STF/results/starfm/syy_setting~9/ML/one_pair~patch_size_120~patch_stride_50~window_size_51~num_classes_20/full/imgs/ML/save_img',
            'FSDAF': '/home/zhaojiazhen/workspace/STF/STF/results/FSDAF/ML/full',
            'FitFC': 'results/FitFC/ML/full/FitFC_Results_L2',
            'stfdcnn': '/home/zhaojiazhen/workspace/STF/STF/results/stfdcnn/syy_setting~9/ML/inference~stage_1_SGD_1e-2~stage_2_SGD_1e-1/full/imgs/stage_2/ML/save_img',
            'ganstfm': '/home/zhaojiazhen/workspace/STF/STF/results/ganstfm/syy_setting-9/ML/inference~Adam_1e-4/full/imgs/ML/save_img',
            'stfgan': 'results/stfgan/syy_setting-9/ML/inference~stage_1~RMSProp~stage_2~RMSProp/full/imgs/stage_2/ML/save_img',
            'opgan': '/home/zhaojiazhen/workspace/STF/STF/results/opgan/syy_setting-9/ML/inference~RMSProp/full/imgs/ML/save_img',
            'swinstf': '/home/zhaojiazhen/workspace/STF/STF/results/swinstf/syy_setting-9/ML/inference~Adam_1e-4~StepLR_step_size-15_gamma_5e-1/full/imgs/ML/save_img',
            'fsdformer': '/home/zhaojiazhen/workspace/STF/STF/results/fsdformer/syy_setting-9/ML/inferencer/full/imgs/ML/save_img',
            'stfmamba': '/home/zhaojiazhen/workspace/STF/STF/results/stfmamba/syy_setting-9/ML/inferencer/full/imgs/ML/save_img',
            'stfdiff': '/home/zhaojiazhen/workspace/STF/STF/results/stfdiff/syy_setting-9/model6_GN_SiLU/ML/inference/full/imgs/ML/0/save_img',
            'GPSTFDiff': '/home/zhaojiazhen/workspace/STF/STF/results/GPSTFDiff/lap/syy_setting-9/ML/inference/full/imgs/ML/0/save_img',
        }
    else:
        print("Error!")
        return


    save_mode = 'both'
    band_config = {'RED': 2, 'NIR': 3}
    # ==================================================

    os.makedirs(out_dir, exist_ok=True)
    true_files = glob.glob(os.path.join(true_dir, '*.tif'))
    if not true_files:
        print(f"在 {true_dir} 中没有找到任何 .tif 文件！")
        return

    methods = list(pred_dirs_dict.keys())

    for true_file in true_files:
        prefix = get_file_prefix(true_file)
        print(f"正在处理数据组: {prefix} ...")
        
        true_data = tiff.imread(true_file)
        true_red_full = extract_band(true_data, band_config['RED'])
        true_nir_full = extract_band(true_data, band_config['NIR'])
        
        # --- 画布搭建：纯色多波段合图 ---
        if save_mode in ['combined', 'both']:
            fig_comb, axes_comb = plt.subplots(nrows=1, ncols=len(methods), figsize=(5 * len(methods), 5))
            if len(methods) == 1: axes_comb = [axes_comb]
            
            # 画布搭建：密度着色（上下两行对应提取的两波段，列为各方法）
            fig_den_comb, axes_den_comb = plt.subplots(nrows=2, ncols=len(methods), figsize=(5 * len(methods), 10))
            if len(methods) == 1: axes_den_comb = np.expand_dims(axes_den_comb, axis=1)
            scatter_den_ref = None # 用于加统一个Colorbar
            
        # --- 画布搭建：单图 ---
        if save_mode in ['individual', 'both']:
            figs_ind, axes_ind = {}, {}
            figs_den_ind, axes_den_ind = {}, {}
            
            for method in methods:
                figs_ind[method], axes_ind[method] = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
                # 密度图：行数固定为二(RED, NIR)，一列
                figs_den_ind[method], axes_den_ind[method] = plt.subplots(nrows=2, ncols=1, figsize=(5, 10))

        # --- 绘图逻辑循环 ---
        for j, method in enumerate(methods):
            method_dir = pred_dirs_dict[method]
            pred_search = glob.glob(os.path.join(method_dir, f"{prefix}*.tif"))
            
            if not pred_search:
                print(f"  [警告] 找不到方法 {method} 对应前缀 {prefix} 的预测文件！跳过。")
                continue
                
            pred_file = pred_search[0]
            pred_data = tiff.imread(pred_file)
            
            pred_red_full = extract_band(pred_data, band_config['RED'])
            pred_nir_full = extract_band(pred_data, band_config['NIR'])
            
            # --- 画纯色合并与单图 ---
            if save_mode in ['combined', 'both']:
                plot_multiband_scatter(axes_comb[j], true_red_full, pred_red_full, true_nir_full, pred_nir_full, 
                                       method_name=method, is_first_col=(j == 0))
            if save_mode in ['individual', 'both']:
                plot_multiband_scatter(axes_ind[method], true_red_full, pred_red_full, true_nir_full, pred_nir_full, 
                                       method_name=method, is_first_col=True)
            
            # --- 画密度图 ---
            if save_mode in ['combined', 'both']:
                scatter_r = plot_density_scatter(axes_den_comb[0, j], true_red_full, pred_red_full, "RED", method, j==0)
                scatter_n = plot_density_scatter(axes_den_comb[1, j], true_nir_full, pred_nir_full, "NIR", method, j==0)
                if scatter_r is not None: scatter_den_ref = scatter_r
                
            if save_mode in ['individual', 'both']:
                sc_r = plot_density_scatter(axes_den_ind[method][0], true_red_full, pred_red_full, "RED", method, True)
                plot_density_scatter(axes_den_ind[method][1], true_nir_full, pred_nir_full, "NIR", method, True)
                if sc_r is not None and scatter_den_ref is None: scatter_den_ref = sc_r

        # --- 保存纯色结果 ---
        if save_mode in ['combined', 'both']:
            fig_comb.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12, wspace=0.15)
            comb_path = os.path.join(out_dir, f'{prefix}_multiband_combined.png')
            fig_comb.savefig(comb_path, dpi=400, bbox_inches='tight', transparent=False, facecolor='white')
            plt.close(fig_comb)

        if save_mode in ['individual', 'both']:
            for method in methods:
                if method in figs_ind:
                    figs_ind[method].subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
                    ind_path = os.path.join(out_dir, f'{prefix}_{method}_multiband_individual.png')
                    figs_ind[method].savefig(ind_path, dpi=400, bbox_inches='tight', transparent=False, facecolor='white')
                    plt.close(figs_ind[method])

        # --- 保存密度着色散点图 (不带 Colorbar) ---
        if save_mode in ['combined', 'both'] and scatter_den_ref:
            fig_den_comb.subplots_adjust(left=0.10, right=0.95, top=0.92, bottom=0.08, wspace=0.15, hspace=0.2)
            den_comb_path = os.path.join(out_dir, f'{prefix}_density_combined.png')
            fig_den_comb.savefig(den_comb_path, dpi=400, bbox_inches='tight', facecolor='white')
            plt.close(fig_den_comb)

        if save_mode in ['individual', 'both']:
            for method in methods:
                if method in figs_den_ind:
                    figs_den_ind[method].subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.08, hspace=0.25)
                    den_ind_path = os.path.join(out_dir, f'{prefix}_{method}_density_individual.png')
                    figs_den_ind[method].savefig(den_ind_path, dpi=400, bbox_inches='tight', facecolor='white')
                    plt.close(figs_den_ind[method])
            
        # --- 单独生成并保存 Colorbar (仅保存一次即可，因为密度映射范围通常一致) ---
        if scatter_den_ref:
            cbar_path = os.path.join(out_dir, f'{prefix}_colorbar_only.png')
            if not os.path.exists(cbar_path):  # 避免对同一前缀重复生成
                fig_cbar, ax_cbar = plt.subplots(figsize=(0.5, 4))
                
                # 新增：构建独立的颜色映射器消除警告
                sm = plt.cm.ScalarMappable(cmap=scatter_den_ref.cmap, norm=scatter_den_ref.norm)
                sm.set_array([])
                cbar = fig_cbar.colorbar(sm, cax=ax_cbar)
                
                ax_cbar.tick_params(labelsize=12)
                fig_cbar.savefig(cbar_path, dpi=400, bbox_inches='tight', facecolor='white')
                plt.close(fig_cbar)

        print(f"  {prefix} 组生成完毕。")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Generate error maps for spatio-temporal fusion results.')
    argparser.add_argument('--dataset', type=str, default='LGC', choices=['CIA', 'LGC', 'ML'], help='Dataset to process (CIA, LGC, or ML)')
    args = argparser.parse_args()
    main(args.dataset)
