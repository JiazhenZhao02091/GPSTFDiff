import os
import tifffile
import numpy as np
import argparse
from tqdm import tqdm

def process_images(input_folder, output_folder):
    """
    Processes all 7-band TIF images in a folder:
    1. Reads original (READ ONLY).
    2. Saves the first 6 bands to NEW output_folder.
    3. Saves the 7th band to NEW output_folder.
    """
    
    # Walk through the directory
    for root, dirs, files in os.walk(input_folder):
        for file in tqdm(files, desc="Processing images"):
            if file.lower().endswith(('.tif', '.tiff')):
                file_path = os.path.join(root, file)
                
                # 计算相对路径，保持子文件夹结构（如果原文件夹里有子文件夹）
                rel_path = os.path.relpath(root, input_folder)
                target_dir = os.path.join(output_folder, rel_path)
                
                # 确保目标文件夹存在
                os.makedirs(target_dir, exist_ok=True)

                target_file_path = os.path.join(target_dir, file)
                
                try:
                    # Read the image using tifffile
                    img = tifffile.imread(file_path)
                    
                    if img.ndim == 3:
                        if img.shape[0] == 7:
                            # (7, H, W)
                            bands_1_6 = img[:6, :, :].copy()
                            band_7 = img[6, :, :].copy()
                            is_channel_first = True
                        elif img.shape[2] == 7:
                            # (H, W, 7)
                            bands_1_6 = img[:, :, :6].copy()
                            band_7 = img[:, :, 6].copy()
                            is_channel_first = False
                        else:
                            print(f"Skipping {file}: Not a 7-band image (shape: {img.shape}).")
                            continue
                            
                        # Construct mask filename
                        file_name_no_ext = os.path.splitext(file)[0]
                        mask_filename = f"mask_{file_name_no_ext}.tif"
                        mask_path = os.path.join(target_dir, mask_filename)
                        
                        # Save to NEW path (Safe, does not overwrite original)
                        tifffile.imwrite(target_file_path, bands_1_6, compression='zlib')
                        
                        # Save the 7th band as mask
                        tifffile.imwrite(mask_path, band_7, compression='zlib')
                        
                    else:
                        print(f"Skipping {file}: Unexpected dimensions {img.ndim}.")

                except Exception as e:
                    print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split 7-band images into 6-band image and 1-band mask.")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing images.")
    parser.add_argument("--output_folder", type=str, default=None, help="Optional: Path to save processed images.")
    
    args = parser.parse_args()
    
    input_dir = args.input_folder
    # 如果没有指定输出目录，自动创建一个带 _processed 后缀的目录
    output_dir = args.output_folder if args.output_folder else os.path.normpath(input_dir) + "_processed"
    
    if os.path.isdir(input_dir):
        print(f"Input Folder:  {input_dir}")
        print(f"Output Folder: {output_dir}")
        process_images(input_dir, output_dir)
        print("Processing complete.")
    else:
        print("Invalid input folder path.")