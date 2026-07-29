import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # 【新增】引入高级颜色控制模块
import matplotlib.font_manager as fm
import math
import matplotlib.cm as cm
import argparse

# ==========================================
# 【新增】顶刊图表全局字体与排版设置
# ==========================================
# 1. 这一步是必须的：确保每次运行都把你的私有字体加入缓存
font_path = os.path.expanduser('~/.local/share/fonts/times.ttf')
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)

plt.rcParams['font.family'] = 'serif'
# 让 matplotlib 按顺序找：优先找 Times，找不到就用 DejaVu Serif (标准的学术平替)
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif']
plt.rcParams['mathtext.fontset'] = 'stix'  # 确保可能出现的数学符号也匹配
plt.rcParams['font.size'] = 14             # 全局基础字号
plt.rcParams['axes.unicode_minus'] = False  # 顺便修复负号显示为方块的问题

def preprocess_to_reflectance(img_array):
    if np.max(img_array) > 1.5:
        return img_array / 10000.0
    return img_array

def generate_error_maps(true_dir, pred_dirs_dict, output_dir, enable_grid_plot=True, cmap='OrRd'):
    # 注意这里默认色带换成了 'OrRd' (白-橙-深红)，视觉冲击力更强
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_extensions = ('.tif', '.tiff', '.png', '.jpg') 
    filenames = [f for f in os.listdir(true_dir) if f.lower().endswith(valid_extensions)]
    
    print(f"共发现 {len(filenames)} 张测试图像，开始生成图表...\n")

    for filename in filenames:
        true_path = os.path.join(true_dir, filename)
        
        try:
            true_img = tifffile.imread(true_path).astype(np.float32)
        except Exception as e:
            continue
            
        true_img = preprocess_to_reflectance(true_img)
        keyword = os.path.splitext(filename)[0]
        keyword_list = keyword.split('_')
        keyword = '_'.join(keyword_list[:2])
        scene_errors = {}
        scene_vmax = 0.0  

        # 1. 计算误差与自适应上限
        for model_name, pred_dir in pred_dirs_dict.items():
            if not os.path.exists(pred_dir): continue
                
            available_files = [f for f in os.listdir(pred_dir) if f.lower().endswith(valid_extensions)]
            matched_pred_file = next((f for f in available_files if keyword in f), None)
            
            if not matched_pred_file: continue
                
            pred_path = os.path.join(pred_dir, matched_pred_file)
            try:
                pred_img = tifffile.imread(pred_path).astype(np.float32)
            except Exception:
                continue

            if true_img.shape != pred_img.shape: continue

            pred_img = preprocess_to_reflectance(pred_img)
            error = np.abs(true_img - pred_img)
            
            if error.ndim == 3:
                error = np.mean(error, axis=0) if error.shape[0] < error.shape[-1] else np.mean(error, axis=-1)
                
            scene_errors[model_name] = error
            
            current_p98 = np.percentile(error, 98)
            if current_p98 > scene_vmax:
                scene_vmax = current_p98

        if not scene_errors:
            continue
        
        print(f"正在处理: {keyword} | Colorbar 上限: {scene_vmax:.4f}")

        # ==========================================
        # 【新增】色彩非线性映射 (Gamma 校正)
        # gamma=1.5 会让低误差区域被强力压制成白色，高误差区域更加醒目
        # 如果觉得背景还不够白，可以把 1.5 改成 1.8 或 2.0
        # ==========================================
        gamma_norm = mcolors.PowerNorm(gamma=1.5, vmin=0, vmax=scene_vmax)

        # 构造一个独立的 mappable 对象
        sm = cm.ScalarMappable(cmap=cmap, norm=gamma_norm)
        sm.set_array([])

        # 2. 绘图
        # ==========================================
        
        # 【新增逻辑】在输出目录下设立一个 "individual" 文件夹，按模型单独保存所有图像
        individual_base_dir = os.path.join(output_dir, 'individual')
        for model_name, error in scene_errors.items():
            model_out_dir = os.path.join(individual_base_dir, model_name)
            if not os.path.exists(model_out_dir):
                os.makedirs(model_out_dir)
                
            # 设置比例贴合原图以防拉伸，如果你希望按实际分辨率保存，也可以调整 figsize
            plt.figure(figsize=(8, 8 * (error.shape[0] / error.shape[1])))
            plt.imshow(error, cmap=cmap, norm=gamma_norm)
            plt.axis('off')
            
            # 【关键修改】去除所有留白和边距
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0,0)
            
            save_path = os.path.join(model_out_dir, f'errormap_{keyword}.png')
            # 使用 pad_inches=0 零边距保存
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=600, transparent=True)
            plt.close()

        # 根据配置决定是否额外保存拼图/网格图
        if enable_grid_plot:
            num_models = len(scene_errors)
            cols = min(4, num_models) 
            rows = math.ceil(num_models / cols)
            
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            if num_models == 1: axes = np.array([axes])
            axes = axes.flatten()

            for idx, (model_name, error) in enumerate(scene_errors.items()):
                ax = axes[idx]
                im = ax.imshow(error, cmap=cmap, norm=gamma_norm)
                ax.set_title(model_name, fontsize=16, fontweight='bold', pad=6) 
                ax.axis('off')

            for idx in range(num_models, len(axes)):
                axes[idx].axis('off')

            plt.tight_layout()
            
            save_path = os.path.join(output_dir, f'Comparison_{keyword}.png')
            plt.savefig(save_path, bbox_inches='tight', dpi=600)
            plt.close()
            
        else:
            for model_name, error in scene_errors.items():
                model_out_dir = os.path.join(output_dir, model_name)
                if not os.path.exists(model_out_dir):
                    os.makedirs(model_out_dir)
                    
                plt.figure(figsize=(8, 6))
                im = plt.imshow(error, cmap=cmap, norm=gamma_norm)
                # (移除原有的 plt.colorbar)
                plt.axis('off')

                save_path = os.path.join(model_out_dir, f'errormap_{keyword}.png')
                plt.savefig(save_path, bbox_inches='tight', dpi=600)
                plt.close()

        # ==========================================
        # 【新增】单独保存该场景的 Colorbar (标尺)
        # ==========================================
        fig_cbar, ax_cbar = plt.subplots(figsize=(0.6, 5))
        cbar = fig_cbar.colorbar(sm, cax=ax_cbar)
        # cbar.set_label('Absolute Error', fontsize=18, labelpad=10)
        cbar.ax.tick_params(labelsize=14) 
        
        cbar_path = os.path.join(output_dir, f'colorbar_{keyword}.png')
        fig_cbar.savefig(cbar_path, bbox_inches='tight', dpi=600, facecolor='white')
        plt.close(fig_cbar)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Generate error maps for spatio-temporal fusion results.')
    argparser.add_argument('--dataset', type=str, default='LGC', choices=['CIA', 'LGC', 'ML'], help='Dataset to process (CIA, LGC, or ML)')
    args = argparser.parse_args()

    if args.dataset == 'CIA':
        TRUE_IMAGES_DIR = 'data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/test/full/Landsat_02'
        OUTPUT_DIR = 'PPT/Errormap/CIA'
        PRED_DIRS_DICT = {
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
    elif args.dataset == 'LGC':
        TRUE_IMAGES_DIR = 'data/spatio_temporal_fusion/LGC/private_data/syy_setting-9/test/full/Landsat_02'
        OUTPUT_DIR = 'PPT/Errormap/LGC'
        PRED_DIRS_DICT = {
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
    elif args.dataset == 'ML':
        TRUE_IMAGES_DIR = 'data/spatio_temporal_fusion/ML/private_data/syy_setting-9/test/full/Landsat_02'
        OUTPUT_DIR = 'PPT/Errormap/ML'
        PRED_DIRS_DICT = {
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
    else:
        raise ValueError("Invalid dataset choice. Please select from 'CIA', 'LGC', or 'ML'.")
    
    generate_error_maps(
        TRUE_IMAGES_DIR, 
        PRED_DIRS_DICT, 
        OUTPUT_DIR, 
        enable_grid_plot=True,   
        cmap='OrRd'  
    )