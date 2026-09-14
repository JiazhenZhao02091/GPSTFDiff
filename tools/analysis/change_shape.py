"""
功能:
    批量检查 tif 图像维度，将通道在前的 CHW 图像转换为 HWC 形状。

参数:
    src_dir (str): 源 tif 图像目录。
    tar_dir (str): 转换后 tif 图像保存目录。

返回:
    None。函数直接写出转换后的影像。

输出:
    在目标目录写出转换或原样保存后的 tif 文件，并打印转换进度。
"""
import tifffile as tiff
import numpy as np
from pathlib import Path
from tqdm import tqdm

def convert_chw_to_hwc(src_dir, tar_dir):
    """
    遍历 src_dir 下的所有 tif 图像，将其从 (C, H, W) 转换为 (H, W, C) 并保存到 tar_dir
    """
    src_path = Path(src_dir)
    tar_path = Path(tar_dir)
    
    # 确保目标文件夹存在
    tar_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有的 tif 文件
    tif_files = list(src_path.glob('*.tif'))
    
    if not tif_files:
        print(f"在目录 {src_dir} 中未找到 .tif 文件！")
        return
        
    for file_path in tqdm(tif_files, desc="转换进度"):
        # 读取图像
        img = tiff.imread(file_path)
        
        # 判断是否为三维图像，并且第一维度像通道数（通常通道数远小于长宽）
        if img.ndim == 3:
            if img.shape[0] < min(img.shape[1], img.shape[2]):
                # 使用 np.transpose 将 (C, H, W) 转换为 (H, W, C)
                # 原始索引 0, 1, 2 对应 C, H, W；目标索引为 1(H), 2(W), 0(C)
                img_hwc = np.transpose(img, (1, 2, 0))
                
                # 保存转换后的图像
                save_path = tar_path / file_path.name
                tiff.imwrite(save_path, img_hwc)
            else:
                # 如果已经是 HWC 或者其他情况，直接复制过去
                save_path = tar_path / file_path.name
                tiff.imwrite(save_path, img)
        else:
             print(f"警告: {file_path.name} 不是3维图像，已跳过转换并直接保存。")
             save_path = tar_path / file_path.name
             tiff.imwrite(save_path, img)
             
    print(f"转换完成！所有文件已保存至: {tar_dir}")

# ================= 使用示例 =================
if __name__ == '__main__':
    # 替换为你实际存放原始 CHW 图像的文件夹路径
    source_directory = "/home/zhaojiazhen/workspace/STF/STF/results/FSDAF/origin/ML/patch"
    
    # 替换为你想要保存转换后(HWC)结果的文件夹路径
    target_directory = "/home/zhaojiazhen/workspace/STF/STF/results/FSDAF/ML/patch"
    
    convert_chw_to_hwc(source_directory, target_directory)
