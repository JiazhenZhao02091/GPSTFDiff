import torch
from model import model_STF

def test_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 1. 实例化模型
    model = model_STF().to(device)
    model.eval()

    # 2. 构造模拟输入数据
    # 根据 train.py，batch_size 为 10，patch_size 为 128
    # 假设通道数为 6 (从 get_features 函数中 image[:,0:3] 和 [:,3:6] 推断)
    batch_size = 1
    channels = 6
    height, width = 256, 256

    ref_lr = torch.randn(batch_size, channels, height, width).to(device)        # C1
    data = torch.randn(batch_size, channels, height, width).to(device)          # C2
    ref_target = torch.randn(batch_size, channels, height, width).to(device)    # F1

    print(f"输入形状: {ref_lr.shape}")

    # 3. 前向传播
    # with torch.no_grad():
    try:
        SR1, SR2, x1, x2, fusion = model(ref_lr, data, ref_target, device)
        
        # 4. 打印输出形状
        print(f"SR1 shape: {SR1.shape}")
        print(f"SR2 shape: {SR2.shape}")
        print(f"x1 shape: {x1.shape}")
        print(f"x2 shape: {x2.shape}")
        print(f"fusion shape: {fusion.shape}")
        
        print("\n模型运行成功！")
    except Exception as e:
        print(f"\n运行出错: {e}")
    out = fusion.mean()
    out.backward()
    print("backward ok")

    
if __name__ == "__main__":
    test_model()