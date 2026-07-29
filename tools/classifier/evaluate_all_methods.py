import os
import re
import sys
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

def generate_legend_log(log_dir, cmap_name=None, vmin=0, vmax=4):
    """生成带有颜色对应的文本信息供对照查看"""
    # 替换为方案一的配色
    cmap = mcolors.ListedColormap(['#90C44B', '#389644', '#EDDEA9', '#0F5724', '#800000'])
    # cmap = mcolors.ListedColormap(['#90C44B', '#389644', '#EDDEA9', '#0F5724', '#A0A0A0'])
    # cmap = mcolors.ListedColormap(['#DEDA78', '#D2CE6E', '#EBEBBE', '#38A800', '#AB0C0C'])
    # cmap = mcolors.ListedColormap(['#FFC20A', '#0C7BDC', '#E66100', '#5D69B1', '#99C945'])
    # cmap = mcolors.ListedColormap(['#DEDA78', '#D2CE6E', '#EBEBBE', '#38A800', '#AB0C0C'])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    legend_file = os.path.join(log_dir, "color_legend.txt")
    with open(legend_file, "w", encoding="utf-8") as f:
        f.write("=== Class Color Legend (RGB) ===\n")
        for class_id in range(vmin, vmax + 1):
            name = CLASS_NAMES.get(class_id, f"Class {class_id}")
            rgba = cmap(norm(class_id))
            r, g, b = [int(x * 255) for x in rgba[:3]]
            f.write(f"ID {class_id:<3} | {name:<12} | RGB: ({r:>3}, {g:>3}, {b:>3})\n")
        # 背景修改为白色
        f.write(f"ID 255 | Background   | RGB: (255, 255, 255) [White]\n")

def save_colored_png(array, save_path, valid_mask=None, cmap_name=None, vmin=0, vmax=4):
    """将类别标签图上色并保存为彩色PNG"""
    # ✅ 修复：已经同步替换为带红色的学术优化版配色！
    # cmap = mcolors.ListedColormap(['#90C44B', '#389644', '#EDDEA9', '#0F5724', '#A0A0A0'])
    cmap = mcolors.ListedColormap(['#90C44B', '#389644', '#EDDEA9', '#0F5724', '#800000'])
    # cmap = mcolors.ListedColormap(['#DEDA78', '#D2CE6E', '#EBEBBE', '#38A800', '#AB0C0C'])
    # cmap = mcolors.ListedColormap(['#FFC20A', '#0C7BDC', '#E66100', '#5D69B1', '#99C945'])
    # cmap = mcolors.ListedColormap(['#DEDA78', '#D2CE6E', '#EBEBBE', '#38A800', '#AB0C0C'])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colored_img = cmap(norm(array))
    rgb_img = (colored_img[:, :, :3] * 255).astype(np.uint8)

    if valid_mask is not None:
        rgb_img[~valid_mask] = [255, 255, 255] # 背景修改为白
    
    img_pil = Image.fromarray(rgb_img)
    img_pil.save(save_path)

def save_confidence_map(conf_array, save_path, valid_mask=None, cmap_name='jet'):
    """
    将置信度图 (0~1) 映射为热力图并保存
    """
    cmap = plt.get_cmap(cmap_name)
    # 置信度通常在 0 (或1/num_classes) 到 1 之间
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    colored_img = cmap(norm(conf_array))
    rgb_img = (colored_img[:, :, :3] * 255).astype(np.uint8)

    if valid_mask is not None:
        rgb_img[~valid_mask] = [255, 255, 255] # 背景修改为白
        
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

def find_pred_file(pred_dir, group_id):
    """模糊匹配寻找预测影像"""
    if not os.path.exists(pred_dir): return None
    for f in os.listdir(pred_dir):
        if group_id in f and f.endswith(('.tif', '.tiff')):
            return os.path.join(pred_dir, f)
    return None

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
        # 通过 Softmax 获取概率分布
        probs = torch.softmax(logits, dim=1) 
        # 获取最大概率及其对应的索引(类别)
        conf, pred = torch.max(probs, dim=1)
        pred = pred.cpu().numpy()[0]
        conf = conf.cpu().numpy()[0]
    return pred, conf


# ================= 核心评估逻辑 =================

def evaluate_method(method_name, pred_dir, cdl_dir, dataset_split, model, device, save_dir):
    print(f"\n{'='*80}")
    print(f"🚀 开始评估模型: {method_name}")
    print(f"{'='*80}")

    method_save_dir = os.path.join(save_dir, method_name)
    os.makedirs(method_save_dir, exist_ok=True)

    # 全局指标收集
    metrics_list = {"oa": [], "kappa": [], "f1_macro": [], "miou": []}
    # 用于计算类别级 F1 和全局混淆矩阵的累加器
    global_cm = np.zeros((len(VALID_LABELS), len(VALID_LABELS)), dtype=np.int64)

    for item in dataset_split:
        group_id = item.get('group_id')
        year = item.get('year')
        
        pred_path = find_pred_file(pred_dir, group_id)
        if not pred_path:
            print(f"[跳过] 找不到 {group_id} 的预测文件，目录: {pred_dir}")
            continue
            
        cdl_filename = next((f for f in os.listdir(cdl_dir) if year in f and f.endswith('.tif')), None)
        if not cdl_filename:
            continue
        cdl_path = os.path.join(cdl_dir, cdl_filename)

        try:
            # 1. 读取真值
            cdl_arr = tiff.imread(cdl_path)
            gt_label = np.full_like(cdl_arr, CONFIG["ignore_label"])
            for k, v in CONFIG["id_map"].items():
                gt_label[cdl_arr == k] = v
            valid_mask = (gt_label != CONFIG["ignore_label"])

            # 2. 读取预测图并推理
            img = preprocess(tiff.imread(pred_path))
            p_map, conf_map = predict_with_confidence(model, img, device)

            # 3. 提取有效区域并计算当前图片的指标
            p = p_map[valid_mask]
            g = gt_label[valid_mask]
            
            if len(g) > 0:
                oa = accuracy_score(g, p)
                kappa = cohen_kappa_score(g, p)
                f1_macro = f1_score(g, p, average='macro', zero_division=0)
                
                cm = confusion_matrix(g, p, labels=VALID_LABELS)
                global_cm += cm # 累加到全局混淆矩阵
                
                intersection = np.diag(cm)
                union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - intersection
                miou = np.mean(intersection / (union + 1e-6))
                
                metrics_list["oa"].append(oa)
                metrics_list["kappa"].append(kappa)
                metrics_list["f1_macro"].append(f1_macro)
                metrics_list["miou"].append(miou)
                
                print(f"[{group_id}] OA: {oa:.4f} | Kappa: {kappa:.4f} | mIoU: {miou:.4f}")

            # 4. 保存可视化结果
            group_save_dir = os.path.join(method_save_dir, group_id)
            os.makedirs(group_save_dir, exist_ok=True)
            
            # 只在第一个模型评估时保存一次 GT 即可，但为了方便对比，这里每个文件夹里都放一个
            save_colored_png(gt_label, os.path.join(group_save_dir, "GT_Class.png"), valid_mask=valid_mask)
            
            # 保存分类结果
            save_colored_png(p_map, os.path.join(group_save_dir, f"{method_name}_Class.png"), valid_mask=valid_mask)
            
            # [新增] 保存分类置信度图
            save_confidence_map(conf_map, os.path.join(group_save_dir, f"{method_name}_Confidence.png"), valid_mask=valid_mask)

        except Exception as e:
            print(f"[{group_id}] 处理发生错误: {e}")

    # ================= 汇总统计与图表绘制 =================
    if len(metrics_list["oa"]) > 0:
        mean_oa = np.mean(metrics_list["oa"])
        mean_kappa = np.mean(metrics_list["kappa"])
        mean_f1_macro = np.mean(metrics_list["f1_macro"])
        mean_miou = np.mean(metrics_list["miou"])

        # [新增] 根据全局混淆矩阵计算类别级 F1 分数
        precision = np.diag(global_cm) / np.maximum(np.sum(global_cm, axis=0), 1)
        recall = np.diag(global_cm) / np.maximum(np.sum(global_cm, axis=1), 1)
        class_f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-8)

        # 打印并保存总结报告
        report = []
        report.append(f"\n{'*'*50}")
        report.append(f"Global Summary for {method_name.upper()}")
        report.append(f"{'*'*50}")
        report.append(f"Average OA:      {mean_oa:.4f}")
        report.append(f"Average Kappa:   {mean_kappa:.4f}")
        report.append(f"Average F1(mac): {mean_f1_macro:.4f}")
        report.append(f"Average mIoU:    {mean_miou:.4f}")
        report.append("\n[Class-wise F1 Scores]")
        for c_idx, c_name in zip(VALID_LABELS, VALID_CLASS_STRINGS):
            report.append(f"  - {c_name:<12}: {class_f1[c_idx]:.4f}")
        report.append(f"{'*'*50}\n")
        
        report_str = "\n".join(report)
        print(report_str)

        with open(os.path.join(method_save_dir, "Metrics_Summary.txt"), "w") as f:
            f.write(report_str)

        # [新增] 绘制并保存混淆矩阵
        cm_save_path = os.path.join(method_save_dir, f"{method_name}_ConfusionMatrix.png")
        plot_confusion_matrix(global_cm, VALID_CLASS_STRINGS, cm_save_path, title=f"Confusion Matrix - {method_name}")
        print(f"✅ 混淆矩阵已保存至: {cm_save_path}")
    else:
        print(f"❌ 模型 {method_name} 没有有效数据计算指标。")


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
    CDL_DIR = "data/spatio_temporal_fusion/ML/CDL_cropped"
    BASE_LOG_DIR = "results/classifier/all_methods_eval"
    os.makedirs(BASE_LOG_DIR, exist_ok=True)

    # 包含所有方法的字典
    METHODS_DICT = {
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
    
    # 初始化全局日志
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(BASE_LOG_DIR, f"global_eval_{LOCATION}_{timestamp}.log")
    sys.stdout = Logger(log_file)
    
    generate_legend_log(BASE_LOG_DIR)
    print(f"=== 全局批量模型推理评估开始，日志保存至: {log_file} ===")

    # 加载分类模型
    model = smp.Unet(
        encoder_name="resnet18",
        in_channels=CONFIG["in_channels"],
        classes=CONFIG["num_classes"],
        encoder_weights=None
    )
    model.load_state_dict(torch.load(CONFIG["model_path"], map_location=CONFIG["device"]))
    model.to(CONFIG["device"])
    
    dataset_split = get_inference_config(LOCATION, split='val')

    if not dataset_split:
        print("⚠️ 警告：从 syy_setting 获取失败，启用内置 fallback 配置...")
        dataset_split = [
            {'group_id': 'Group_01', 'year': '2021'},
            {'group_id': 'Group_02', 'year': '2022'},
            {'group_id': 'Group_03', 'year': '2023'},
            {'group_id': 'Group_04', 'year': '2023'} # 年份请根据你实际的 CDL 年份修改
        ]
    # 遍历所有的模型方法
    for method_name, pred_dir in METHODS_DICT.items():
        evaluate_method(
            method_name=method_name,
            pred_dir=pred_dir,
            cdl_dir=CDL_DIR,
            dataset_split=dataset_split,
            model=model,
            device=CONFIG["device"],
            save_dir=BASE_LOG_DIR
        )
            
    print("\n🎉 所有模型评估已全部完成！")
