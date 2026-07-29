import torch
import tifffile as tiff
import numpy as np
import segmentation_models_pytorch as smp
from tools.classifier.config import CONFIG

def predict(img_path, output_path):
    # 加载模型
    model = smp.Unet(
        encoder_name="resnet18",
        in_channels=CONFIG["in_channels"],
        classes=CONFIG["num_classes"],
        encoder_weights=None
    ).to(CONFIG["device"])
    model.load_state_dict(torch.load(CONFIG["model_path"], map_location=CONFIG["device"]))
    model.eval()

    # 读取并预处理
    img_raw = tiff.imread(img_path).astype(np.float32)
    # 生成有效数据掩膜：如果 6 个波段全是 0，则该像素为背景
    data_mask = np.sum(img_raw, axis=2) > 0 
    
    img_input = (img_raw / 10000.0).transpose(2, 0, 1)
    img_input = torch.from_numpy(img_input).unsqueeze(0).to(CONFIG["device"])

    with torch.no_grad():
        output = model(img_input)
        # 获取概率最大的索引 (0-4)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # 还原 CDL 原始 ID
    result = np.zeros(pred.shape, dtype=np.uint8)
    for train_id, raw_id in CONFIG["rev_map"].items():
        result[pred == train_id] = raw_id
    
    # 强制将原始无效区设为 0
    result[~data_mask] = 0

    tiff.imwrite(output_path, result)
    print(f"预测完成，结果保存至: {output_path}")

if __name__ == "__main__":
    predict("./data/test/landsat_example.tif", "classification_result.tif")
