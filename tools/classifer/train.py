import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset, random_split
import tifffile as tiff
import numpy as np
import os
import re
from config import CONFIG
from sklearn.metrics import confusion_matrix

class LandsatDataset(Dataset):

    def __init__(self, img_dir, mask_dir):
        self.img_ids = sorted([f for f in os.listdir(img_dir) if f.endswith('.tif')])
        self.img_dir = img_dir
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.tif')])
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.img_ids)

    def preprocess_mask(self, mask):
        target = np.full(mask.shape, CONFIG['ignore_label'], dtype=np.int64)
        for raw_id, train_id in CONFIG['id_map'].items():
            target[mask == raw_id] = train_id
        return target

    def __getitem__(self, idx):
        img_name = self.img_ids[idx]
        year_match = re.search('(\\d{4})', img_name)
        mask_name = None
        if year_match:
            year = year_match.group(1)
            for f in self.mask_files:
                if year in f:
                    mask_name = f
                    break
        if mask_name is None:
            raise FileNotFoundError(f"Could not find a mask file matching year {(year if year_match else 'unknown')} for image {img_name}")
        img = tiff.imread(os.path.join(self.img_dir, img_name)).astype(np.float32)
        mask = tiff.imread(os.path.join(self.mask_dir, mask_name))
        img = img.transpose(2, 0, 1)
        target = self.preprocess_mask(mask)
        return (torch.from_numpy(img), torch.from_numpy(target))

def validate(model, loader, device, num_classes, ignore_label):
    model.eval()
    total_samples = 0
    correct_samples = 0
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            preds_flat = preds.view(-1).cpu().numpy()
            masks_flat = masks.view(-1).cpu().numpy()
            valid_mask = masks_flat != ignore_label
            p = preds_flat[valid_mask]
            t = masks_flat[valid_mask]
            if len(t) == 0:
                continue
            correct_samples += np.sum(p == t)
            total_samples += len(t)
            for i in range(num_classes):
                pred_i = p == i
                true_i = t == i
                intersection[i] += np.logical_and(pred_i, true_i).sum()
                union[i] += np.logical_or(pred_i, true_i).sum()
    oa = correct_samples / total_samples if total_samples > 0 else 0
    iou_per_class = intersection / (union + 1e-06)
    miou = np.mean(iou_per_class)
    return (oa, miou)

def train():
    full_dataset = LandsatDataset(CONFIG['Landsat_path'], CONFIG['CDL_path'])
    metric_info = CONFIG['info_name']
    val_percent = 0.2
    n_val = int(len(full_dataset) * val_percent)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    print(f'Dataset Split: Train={len(train_dataset)}, Val={len(val_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)
    model = smp.Unet(encoder_name='resnet18', in_channels=CONFIG['in_channels'], classes=CONFIG['num_classes'], encoder_weights=None).to(CONFIG['device'])
    best_miou = 0.0
    weights = torch.tensor([1.0, 1.0, 2.5, 3.0, 3.5]).to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=CONFIG['ignore_label'])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        for imgs, masks in train_loader:
            imgs, masks = (imgs.to(CONFIG['device']), masks.to(CONFIG['device']))
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step_count += 1
        train_avg_loss = epoch_loss / step_count
        print(f'Epoch {epoch + 1} Train Loss: {train_avg_loss:.4f}...', end=' ')
        val_oa, val_miou = validate(model, val_loader, CONFIG['device'], CONFIG['num_classes'], CONFIG['ignore_label'])
        print(f'Val OA: {val_oa:.4f} | Val mIoU: {val_miou:.4f}')
        if val_miou > best_miou:
            print(f'  >>> New Best mIoU! ({best_miou:.4f} -> {val_miou:.4f}). Saving model...')
            best_miou = val_miou
            torch.save(model.state_dict(), CONFIG['model_path'])
            with open(metric_info, 'w') as f:
                f.write(f'Epoch: {epoch + 1}, mIoU: {val_miou}, OA: {val_oa}')
    print(f'Training finished. Best Val mIoU was: {best_miou:.4f}')
if __name__ == '__main__':
    train()
