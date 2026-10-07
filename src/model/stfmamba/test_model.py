import torch
from model import model_STF

def test_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    model = model_STF().to(device)
    model.eval()


    batch_size = 1
    channels = 6
    height, width = 256, 256

    ref_lr = torch.randn(batch_size, channels, height, width).to(device)
    data = torch.randn(batch_size, channels, height, width).to(device)
    ref_target = torch.randn(batch_size, channels, height, width).to(device)

    print(f"输入形状: {ref_lr.shape}")


    try:
        SR1, SR2, x1, x2, fusion = model(ref_lr, data, ref_target, device)


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
