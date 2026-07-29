import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset, random_split
import tifffile as tiff
import numpy as np
import os
import re
from tools.classifier.config import CONFIG
from sklearn.metrics import confusion_matrix # 用于计算 IoU 和 Kappa

"""
    * 相较与train.py的改动 通过数据集来计算 loss 的权重，而不是之前的手动设置一个固定的权重数组。
"""


class LandsatDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_ids = sorted([f for f in os.listdir(img_dir) if f.endswith('.tif')])
        self.img_dir = img_dir
        
        # 获取 mask 目录下所有的文件，不再要求一一对应，而是按年份匹配
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.tif')])
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.img_ids)

    def preprocess_mask(self, mask):
        # 初始化为忽略值
        target = np.full(mask.shape, CONFIG["ignore_label"], dtype=np.int64)

        for raw_id, train_id in CONFIG["id_map"].items():
            target[mask == raw_id] = train_id
        # 对于既不是背景0，又没在ID_MAP中的其他零星像素
        # 直接忽略 (保持 255)

        return target

    def __getitem__(self, idx):
        img_name = self.img_ids[idx]
        
        # 提取图像文件名中的年份 (4位数字)
        year_match = re.search(r'(\d{4})', img_name)
        mask_name = None
        
        if year_match:
            year = year_match.group(1)
            # 在 mask 文件列表中查找包含该年份的文件
            for f in self.mask_files:
                if year in f:
                    mask_name = f
                    break
        
        if mask_name is None:
            raise FileNotFoundError(f"Could not find a mask file matching year {year if year_match else 'unknown'} for image {img_name}")

        img = tiff.imread(os.path.join(self.img_dir, img_name)).astype(np.float32)
        mask = tiff.imread(os.path.join(self.mask_dir, mask_name))
        
        # 归一化与维度转换 [C, H, W]
        # img = (img / 10000.0).transpose(2, 0, 1)
        img = img.transpose(2, 0, 1)
        target = self.preprocess_mask(mask)
        
        return torch.from_numpy(img), torch.from_numpy(target)

# 新增：一个计算验证集指标的函数
def validate(model, loader, device, num_classes, ignore_label):
    model.eval() # 切换到评估模式 (关闭 Dropout, BN 使用移动平均)
    
    # 初始化混淆矩阵
    total_samples = 0
    correct_samples = 0
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    
    with torch.no_grad(): # 验证时不计算梯度，省显存
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            outputs = model(imgs) # [B, Num_Classes, H, W]
            preds = torch.argmax(outputs, dim=1) # [B, H, W]
            
            # 展平以便通过 mask 过滤
            preds_flat = preds.view(-1).cpu().numpy()
            masks_flat = masks.view(-1).cpu().numpy()
            
            # 过滤掉 ignore_label
            valid_mask = (masks_flat != ignore_label)
            p = preds_flat[valid_mask]
            t = masks_flat[valid_mask]
            
            if len(t) == 0:
                continue

            # 使用 sklearn 计算混淆矩阵 (或者手动写，这里示范逻辑)
            # 为了速度，可以在 GPU 上计算简单的 IoU，这里写个简单逻辑：
            
            # 计算 Accuracy
            correct_samples += np.sum(p == t)
            total_samples += len(t)
            
            # 计算 IoU (逐类)
            for i in range(num_classes):
                # 预测是 i 的集合
                pred_i = (p == i)
                # 真实是 i 的集合
                true_i = (t == i)
                
                intersection[i] += np.logical_and(pred_i, true_i).sum()
                union[i] += np.logical_or(pred_i, true_i).sum()

    # 指标聚合
    oa = correct_samples / total_samples if total_samples > 0 else 0
    
    # 计算 mIoU (注意处理分母为0的情况)
    iou_per_class = intersection / (union + 1e-6)
    miou = np.mean(iou_per_class)
    
    return oa, miou

# 新增：自动计算数据集中各类别权重的函数
def calculate_class_weights(dataset, num_classes, ignore_label):
    print("正在统计训练集像素以计算 Loss 权重...")
    class_counts = np.zeros(num_classes, dtype=np.int64)
    
    for _, mask in dataset:
        mask_np = mask.numpy()
        valid_mask = mask_np != ignore_label
        labels = mask_np[valid_mask]
        if len(labels) > 0:
            counts = np.bincount(labels, minlength=num_classes)
            class_counts += counts

    # 避免除以 0
    class_counts = np.maximum(class_counts, 1)
    
    # 计算频率
    frequencies = class_counts / np.sum(class_counts)
    # 中值频率平衡 (Median Frequency Balancing)
    median_freq = np.median(frequencies)
    weights = median_freq / frequencies
    
    # 限制权重的最大值，防止极小类别的权重爆炸导致梯度不稳定
    weights = np.clip(weights, 0.1, 10.0)
    
    print(f"像素统计: {class_counts}")
    print(f"自动计算的权重: {np.round(weights, 4)}")
    return torch.tensor(weights, dtype=torch.float32)

def train():
    os.makedirs(os.path.dirname(CONFIG["model_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["info_name"]), exist_ok=True)

    # 1. 准备数据集
    full_dataset = LandsatDataset(CONFIG["Landsat_path"], CONFIG["CDL_path"])

    metric_info = CONFIG["info_name"]
    
    # 【新增】划分训练集和验证集 (8:2)
    val_percent = 0.2
    n_val = int(len(full_dataset) * val_percent)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    
    print(f"Dataset Split: Train={len(train_dataset)}, Val={len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)

    # 2. 模型
    model = smp.Unet(
        encoder_name="resnet18",
        in_channels=CONFIG["in_channels"],
        classes=CONFIG["num_classes"],
        encoder_weights=None # 如果数据是多光谱非RGB，不要用imagenet预训练权重，或者修一层
    ).to(CONFIG["device"])

    # 3. 优化器与 Loss
    best_miou = 0.0 
    
    # 【修改】使用自动计算的权重替换硬编码权重
    # weights = torch.tensor([1.0, 1.0, 2.5, 3.0, 3.5]).to(CONFIG["device"])
    weights = calculate_class_weights(train_dataset, CONFIG["num_classes"], CONFIG["ignore_label"]).to(CONFIG["device"])
    
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=CONFIG["ignore_label"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    for epoch in range(CONFIG["epochs"]):
        # --- Training Loop ---
        model.train()
        epoch_loss = 0.0 
        step_count = 0
        
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(CONFIG["device"]), masks.to(CONFIG["device"])
            
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            step_count += 1
        
        train_avg_loss = epoch_loss / step_count
        
        # --- Validation Loop (每个 Epoch 结束做一次) ---
        print(f"Epoch {epoch+1} Train Loss: {train_avg_loss:.4f}...", end=" ")
        
        val_oa, val_miou = validate(
            model, val_loader, CONFIG["device"], 
            CONFIG["num_classes"], CONFIG["ignore_label"]
        )
        
        print(f"Val OA: {val_oa:.4f} | Val mIoU: {val_miou:.4f}")

        # 【修改点】仅在 验证集 mIoU 创新高时保存模型
        if val_miou > best_miou:
            print(f"  >>> New Best mIoU! ({best_miou:.4f} -> {val_miou:.4f}). Saving model...")
            best_miou = val_miou
            torch.save(model.state_dict(), CONFIG["model_path"]) 
            # 保存一下对应的 OA 以便记录
            with open(metric_info, "w") as f:
                f.write(f"Epoch: {epoch+1}, mIoU: {val_miou}, OA: {val_oa}")

    print(f"Training finished. Best Val mIoU was: {best_miou:.4f}")

if __name__ == "__main__":
    train()
