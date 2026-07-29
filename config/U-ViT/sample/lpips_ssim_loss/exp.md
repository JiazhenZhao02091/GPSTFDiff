# 实验编号: [EXP 001]

**实验名称:** [修改U-ViT的损失函数为 l1 + SSIM + LPIPS ]

**日期:** 2025-11-09

## 1. 🎯 实验目标 (Hypothesis)

* [测试更换损失函数是否对最终的定量指标有所优化]

## 2. 🛠️ 核心配置 (Configuration)

### A. 模型 (Model)
* **架构:** [基于U-ViT的放STFDIFF的双流编码]
<!-- * **预训练:** [例如: ImageNet-22k (如果使用了)] -->
* **修改点:** [修改了损失函数]

### B. 数据集 (Dataset)
* **名称:** [例如: CIA/LGC 数据集]
* **训练集大小:** [例如: 10,000 图像对]
* **验证集大小:** [例如: 2,000 图像对]
* **数据增强:** [例如: 随机翻转、随机裁剪 (256x256)]

### C. 超参数 (Hyperparameters)
* **Batch Size:** 32
* **Learning Rate (LR):** 1e-4
* **优化器 (Optimizer):** AdamW
* **权重衰减 (Weight Decay):** 0.05
* **Epochs:** 100
* **损失函数 (Loss):** L1 Loss + SSIM Loss + LPIPS Loss 其中权重分别为 0.8 0.2 0.05

### D. 环境 (Environment)
* **GPU:** [例如: 1x NVIDIA RTX 4090]
* **PyTorch 版本:** 2.0.1
* **CUDA 版本:** 11.8

## 3. 📈 实验过程 (Progress Log)

* **[可选] 关联的日志文件:** `logs/exp-001/training.log`
* **[可选] 关联的权重:** `checkpoints/exp-001/best_model.pth`

* **Epoch 10/100:**
    * Train Loss: 0.152
    * Val Loss: 0.130
    * Val mIoU: 0.65
* **Epoch 20/100:**
    * Train Loss: 0.110
    * Val Loss: 0.105
    * Val mIoU: 0.72
* ...
* **Epoch 80/100:**
    * **[发现]** 模型在验证集上出现过拟合，Val Loss 开始上升。

## 4. 📊 最终结果 (Results)

* **最佳 Epoch:** 75
* **最佳验证集 mIoU:** 0.81
* **测试集 mIoU:** 0.80

## 5. 🧠 结论与分析 (Analysis & Next Steps)

* **结论:** [例如：DINOv3 基线效果良好，但在 75-80 epoch 左右开始过拟合。]
* **遇到的问题:** [例如：L1 Loss 导致边缘模糊，SSIM 的加入有所改善。]
* **下一步计划:**
    1.  [例如：尝试更强的正则化或 Early Stopping。]
    2.  [例如：将 LR 降低到 1e-5 进行微调。]
    3.  [例如：在 Encoder 中加入频域过滤模块 (FFFM) 进行对比实验 (EXP-002)。]