import os
import re
import sys
import glob
import datetime
import numpy as np
import torch
import tifffile as tiff
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, confusion_matrix
import segmentation_models_pytorch as smp

from tools.classifier.config import CONFIG

# 尝试导入 syy_setting
try:
    from scripts.spatio_temparol_fusion.dataset_generation.dataset_config.syy_setting import DATASET_SETTING
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    try:
        from scripts.spatio_temparol_fusion.dataset_generation.dataset_config.syy_setting import DATASET_SETTING
    except ImportError:
        print("Warning: Could not import DATASET_SETTING from syy_setting.py")
        DATASET_SETTING = {}

# 预设的分类名称映射 (0-4) 和背景 (255)
CLASS_NAMES = {
    0: "Corn",
    1: "Soybean",
    2: "Grassland",
    3: "Forest",
    4: "Developed",
    255: "Background"
}
# 用于混淆矩阵的标签列表
VALID_LABELS = [0, 1, 2, 3, 4]
VALID_CLASS_STRINGS = [CLASS_NAMES[i] for i in VALID_LABELS]

# ================= 工具函数 =================

def generate_legend_log(log_dir, cmap_name='tab20', vmin=0, vmax=4):
    """生成带有颜色对应的文本信息供对照查看"""
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    legend_file = os.path.join(log_dir, "color_legend.txt")
    with open(legend_file, "w", encoding="utf-8") as f:
        f.write("=== Class Color Legend (RGB) ===\n")
        for class_id in range(vmin, vmax + 1):
            name = CLASS_NAMES.get(class_id, f"Class {class_id}")
            rgba = cmap(norm(class_id))
            r, g, b = [int(x * 255) for x in rgba[:3]]
            f.write(f"ID {class_id:<3} | {name:<12} | RGB: ({r:>3}, {g:>3}, {b:>3})\n")
        f.write(f"ID 255 | Background   | RGB: (  0,   0,   0) [Black]\n")

def save_colored_png(array, save_path, valid_mask=None, cmap_name='tab20', vmin=0, vmax=4):
    """将类别标签图上色并保存为彩色PNG"""
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colored_img = cmap(norm(array))
    rgb_img = (colored_img[:, :, :3] * 255).astype(np.uint8)

    if valid_mask is not None:
        rgb_img[~valid_mask] = [0, 0, 0] # 背景设为黑
    
    img_pil = Image.fromarray(rgb_img)
    img_pil.save(save_path)

def save_confidence_map(conf_array, save_path, valid_mask=None, cmap_name='jet'):
    """
    将置信度图 (0~1) 映射为热力图并保存
    """
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    colored_img = cmap(norm(conf_array))
    rgb_img = (colored_img[:, :, :3] * 255).astype(np.uint8)

    if valid_mask is not None:
        rgb_img[~valid_mask] = [0, 0, 0] # 背景设为黑
        
    img_pil = Image.fromarray(rgb_img)
    img_pil.save(save_path)

def plot_confusion_matrix(cm, classes, save_path, title='Confusion Matrix'):
    """绘制并保存混淆矩阵热力图"""
    # 归一化混淆矩阵
    cm_norm = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 在格子中填写数值
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def get_year_from_filename(filename):
    match = re.search(r'(\d{4})', filename)
    return match.group(1) if match else None

def get_inference_config(location, split='val', setting_dict=DATASET_SETTING):
    if location not in setting_dict:
        return []
    triplets = setting_dict[location].get(split, [])
    dataset_split = []
    for idx, triplet in enumerate(triplets):
        target_date = triplet[1] if len(triplet) >= 2 else (triplet[0] if len(triplet) == 1 else None)
        if not target_date: continue
        dataset_split.append({
            'group_id': f"Group_{idx+1:02d}",
            'date': target_date,
            'year': get_year_from_filename(target_date) or target_date.split('-')[0]
        })
    return dataset_split

def preprocess(img):
    if img.ndim == 3:
        if img.shape[2] < img.shape[0]: 
            img = np.transpose(img, (2, 0, 1))
    elif img.ndim == 2:
        img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    return img

def predict_with_confidence(model, img, device):
    """同时返回预测类别和分类置信度"""
    model.eval()
    model.to(device)
    tensor_img = torch.from_numpy(img).to(device)
    with torch.no_grad():
        x = tensor_img.unsqueeze(0)
        logits = model(x)
        probs = torch.softmax(logits, dim=1) 
        conf, pred = torch.max(probs, dim=1)
        pred = pred.cpu().numpy()[0]
        conf = conf.cpu().numpy()[0]
    return pred, conf


# ================= 核心评估逻辑 =================

def evaluate_method_patch(method_name, pred_dir, cdl_dir, dataset_split, model, device, save_dir):
    """
    Patch 级别的评估逻辑。
    会在 pred_dir 中查找符合 Group 的切片，并根据切片的坐标在 CDL 目录中匹配 Ground Truth。
    """
    print(f"\n{'='*80}")
    print(f"🚀 开始评估模型 (Patch Mode): {method_name}")
    print(f"{'='*80}")

    method_save_dir = os.path.join(save_dir, method_name)
    os.makedirs(method_save_dir, exist_ok=True)

    # 全局指标累加器
    metrics_list = {"oa": [], "kappa": [], "f1_macro": [], "miou": []}
    global_cm = np.zeros((len(VALID_LABELS), len(VALID_LABELS)), dtype=np.int64)

    total_patches_evaluated = 0

    for item in dataset_split:
        group_id = item.get('group_id')
        year = item.get('year')
        
        # 1. 查找当前 Group 的所有预测 Patch
        # 例如: Group_01_save_img_0_256_0_256.tif
        search_pattern = os.path.join(pred_dir, f"{group_id}_*.tif*")
        pred_patch_files = glob.glob(search_pattern)
        
        if not pred_patch_files:
            print(f"[跳过] 找不到 {group_id} 的任何预测 Patch 文件，搜索模式: {search_pattern}")
            continue
            
        print(f"    找到 {len(pred_patch_files)} 个属于 {group_id} 的 Patch。")
        
        for pred_file in pred_patch_files:
            pred_base = os.path.splitext(os.path.basename(pred_file))[0]
            
            # 2. 从预测文件名中解析出坐标部分
            # 假设你的预测文件形如: Group_01_save_img_0_256_0_256
            # 我们需要提取出 "0_256_0_256"
            parts = pred_base.split('_')
            # 找到包含坐标的连续四个数字部分 (y1_y2_x1_x2)
            coords = None
            if len(parts) >= 4:
                coords = '_'.join(parts[-4:])
                
            if not coords:
                print(f"[警告] 无法从预测文件名 {pred_base} 中解析出坐标，跳过。")
                continue

            # 3. 构建对应的 CDL Patch 文件名
            # 根据你的截图，CDL Patch 命名如: CDL2023_0_256_0_256.tif
            cdl_filename = f"CDL{year}_{coords}.tif"
            cdl_path = os.path.join(cdl_dir, cdl_filename)
            
            if not os.path.exists(cdl_path):
                print(f"[警告] 找不到对应的 GT 文件: {cdl_path}")
                continue

            # 4. 读取真值与预测值，进行推理和指标计算
            try:
                # 读取 CDL
                cdl_arr = tiff.imread(cdl_path)
                gt_label = np.full_like(cdl_arr, CONFIG["ignore_label"])
                for k, v in CONFIG["id_map"].items():
                    gt_label[cdl_arr == k] = v
                valid_mask = (gt_label != CONFIG["ignore_label"])

                # 如果这个 Patch 里全是无效像素 (Background)，直接跳过，不参与统计
                if not np.any(valid_mask):
                    continue

                # 读取预测图并推理
                img = preprocess(tiff.imread(pred_file))
                p_map, conf_map = predict_with_confidence(model, img, device)

                # 提取有效区域
                p = p_map[valid_mask]
                g = gt_label[valid_mask]
                
                if len(g) > 0:
                    total_patches_evaluated += 1
                    oa = accuracy_score(g, p)
                    kappa = cohen_kappa_score(g, p)
                    f1_macro = f1_score(g, p, average='macro', zero_division=0)
                    
                    cm = confusion_matrix(g, p, labels=VALID_LABELS)
                    global_cm += cm 
                    
                    intersection = np.diag(cm)
                    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - intersection
                    miou = np.mean(intersection / (union + 1e-6))
                    
                    metrics_list["oa"].append(oa)
                    metrics_list["kappa"].append(kappa)
                    metrics_list["f1_macro"].append(f1_macro)
                    metrics_list["miou"].append(miou)
                    
                # 5. 可选：保存每个 Patch 的可视化结果（可能会生成大量图片，建议按需开启）
                # 如果不需要保存每一个小块的图片，可以注释掉以下几行
                patch_save_dir = os.path.join(method_save_dir, group_id)
                os.makedirs(patch_save_dir, exist_ok=True)
                
                # save_colored_png(gt_label, os.path.join(patch_save_dir, f"GT_{coords}.png"), valid_mask=valid_mask)
                # save_colored_png(p_map, os.path.join(patch_save_dir, f"Pred_{coords}.png"), valid_mask=valid_mask)
                # save_confidence_map(conf_map, os.path.join(patch_save_dir, f"Conf_{coords}.png"), valid_mask=valid_mask)

            except Exception as e:
                print(f"[{pred_base}] 处理发生错误: {e}")

    # ================= 汇总统计与图表绘制 =================
    if total_patches_evaluated > 0:
        mean_oa = np.mean(metrics_list["oa"])
        mean_kappa = np.mean(metrics_list["kappa"])
        mean_f1_macro = np.mean(metrics_list["f1_macro"])
        mean_miou = np.mean(metrics_list["miou"])

        precision = np.diag(global_cm) / np.maximum(np.sum(global_cm, axis=0), 1)
        recall = np.diag(global_cm) / np.maximum(np.sum(global_cm, axis=1), 1)
        class_f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-8)

        report = []
        report.append(f"\n{'*'*50}")
        report.append(f"Global Summary for {method_name.upper()} (Evaluated {total_patches_evaluated} patches)")
        report.append(f"{'*'*50}")
        report.append(f"Average Patch OA:      {mean_oa:.4f}")
        report.append(f"Average Patch Kappa:   {mean_kappa:.4f}")
        report.append(f"Average Patch F1(mac): {mean_f1_macro:.4f}")
        report.append(f"Average Patch mIoU:    {mean_miou:.4f}")
        report.append("\n[Global Class-wise F1 Scores]")
        for c_idx, c_name in zip(VALID_LABELS, VALID_CLASS_STRINGS):
            report.append(f"  - {c_name:<12}: {class_f1[c_idx]:.4f}")
        report.append(f"{'*'*50}\n")
        
        report_str = "\n".join(report)
        print(report_str)

        with open(os.path.join(method_save_dir, "Metrics_Summary.txt"), "w") as f:
            f.write(report_str)

        cm_save_path = os.path.join(method_save_dir, f"{method_name}_ConfusionMatrix.png")
        plot_confusion_matrix(global_cm, VALID_CLASS_STRINGS, cm_save_path, title=f"Confusion Matrix - {method_name}")
        print(f"✅ 混淆矩阵已保存至: {cm_save_path}")
    else:
        print(f"❌ 模型 {method_name} 没有成功评估任何有效 Patch。")


# ================= 日志记录器 =================
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ================= 主程序入口 =================
if __name__ == "__main__":
    LOCATION = "ML" 
    
    # 【注意】如果是评估 Patch，这里的 CDL 目录必须指向存放 256x256 CDL 切片的地方
    CDL_DIR = "data/spatio_temporal_fusion/ML/CDL_patch"
    
    BASE_LOG_DIR = "results/classifier/all_methods_eval_patch"
    os.makedirs(BASE_LOG_DIR, exist_ok=True)

    # 包含所有方法的字典 (指向各个模型的 patch 预测目录)
    METHODS_DICT = {
        'swinstf': 'results/swinstf/syy_setting-9/ML/inference~Adam_1e-4~StepLR_step_size-15_gamma_5e-1/patch/imgs/ML/save_img',
        # 你可以在这里加入其他模型的 /patch/imgs/ 目录进行公平的 Patch-wise 比较
    }
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(BASE_LOG_DIR, f"global_eval_{LOCATION}_{timestamp}.log")
    sys.stdout = Logger(log_file)
    
    generate_legend_log(BASE_LOG_DIR)
    print(f"=== 全局批量模型推理评估 (Patch Mode) 开始，日志保存至: {log_file} ===")

    # 加载分类模型
    model = smp.Unet(
        encoder_name="resnet18",
        in_channels=CONFIG["in_channels"],
        classes=CONFIG["num_classes"],
        encoder_weights=None
    )
    model.load_state_dict(torch.load(CONFIG["model_path"], map_location=CONFIG["device"]))
    model.to(CONFIG["device"])
    
    dataset_split = get_inference_config(LOCATION, split='test') # 这里注意检查是 'test' 还是 'val'

    if not dataset_split:
        print("⚠️ 警告：从 syy_setting 获取失败，启用内置 fallback 配置...")
        dataset_split = [
            {'group_id': 'Group_01', 'year': '2021'},
            {'group_id': 'Group_02', 'year': '2022'},
            {'group_id': 'Group_03', 'year': '2023'},
            {'group_id': 'Group_04', 'year': '2023'} # 年份请根据你实际的 CDL 年份修改
        ]
    # else:
    # 遍历所有的模型方法
    for method_name, pred_dir in METHODS_DICT.items():
        evaluate_method_patch(
            method_name=method_name,
            pred_dir=pred_dir,
            cdl_dir=CDL_DIR,
            dataset_split=dataset_split,
            model=model,
            device=CONFIG["device"],
            save_dir=BASE_LOG_DIR
        )
            
    print("\n🎉 所有模型评估已全部完成！")
