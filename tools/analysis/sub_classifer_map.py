"""
功能:
    对分类结果 PNG 按 Group 和模型目录批量裁剪 patch，并保存带框 full_marked 图。

参数:
    --input_dir (str): 模型结果父目录，可包含多个模型子目录。
    --output_dir (str): 裁切结果保存根目录。
    --bboxes (list[str]): 一个或多个 bbox 字符串，格式为 y0,y1,x0,x1。
    --filter (str): 可选的模型子目录名称过滤字符串。

返回:
    None。函数直接保存图片文件。

输出:
    在输出目录按模型和 Group 保存 patch PNG 与 full_marked PNG。
"""
import os
import argparse
import shutil
import cv2

# 限定只处理 PNG 图片
VALID_EXTS = ('.png',)

def process_and_save_png(src_path, out_group_dir, prefix, bboxes):
    """
    专门针对 PNG 图像的裁切和标记函数
    """
    # 1. 读取图像 (cv2 默认读取为 BGR)
    img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法读取图像 (可能损坏或不存在): {src_path}")

    # 2. 保存一份纯净的原图副本 (可选，这里命名为 _full)
    # shutil.copy(src_path, os.path.join(out_group_dir, f"{prefix}_full.png"))

    # 3. 创建带有标记框的图像副本
    img_marked = img.copy()
    
    # 确保图像有三个通道以便画彩色框 (如果是单通道灰度图，转为BGR)
    if len(img_marked.shape) == 2:
        img_marked = cv2.cvtColor(img_marked, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # patch 也保持三通道
    
    # BGR 格式的红色边框，线宽 4
    box_color = (0, 0, 255) 
    thickness = 4

    # 4. 遍历并裁切 BBox
    for idx, bbox in enumerate(bboxes):
        y0, y1, x0, x1 = bbox
        
        # 边界保护，防止 bbox 越界
        h, w = img.shape[:2]
        y0, y1 = max(0, y0), min(h, y1)
        x0, x1 = max(0, x0), min(w, x1)

        # 提取切片
        patch = img[y0:y1, x0:x1]
        
        # 命名格式: 原名_patch_1_(y0,y1,x0,x1).png
        patch_name = f"{prefix}_patch_{idx+1}_{y0}_{y1}_{x0}_{x1}.png"
        cv2.imwrite(os.path.join(out_group_dir, patch_name), patch)

        # 在 full_marked 上绘制矩形框
        cv2.rectangle(img_marked, (x0, y0), (x1, y1), box_color, thickness)
        
        # （可选）在框的左上角打上编号标签，方便与切片对应
        # label = f"P{idx+1}"
        # cv2.putText(img_marked, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, box_color, 3)

    # 5. 保存带框的完整图像
    marked_name = f"{prefix}_full_marked.png"
    cv2.imwrite(os.path.join(out_group_dir, marked_name), img_marked)


def crop_groups_folder(input_root, output_root, crop_bboxes_list, ext_filters=VALID_EXTS):
    """
    遍历 input_root 下的 Group_* 子目录，对每个组内的图片裁切并保存到 output_root。
    """
    os.makedirs(output_root, exist_ok=True)

    entries = sorted(os.listdir(input_root))
    for entry in entries:
        entry_path = os.path.join(input_root, entry)
        
        # 只处理目录（如 Group_01, Group_02 等）
        if os.path.isdir(entry_path):
            group_name = entry
            group_input_dir = entry_path
        else:
            continue

        out_group_dir = os.path.join(output_root, group_name)
        os.makedirs(out_group_dir, exist_ok=True)

        # 收集 PNG 图片
        files = [f for f in sorted(os.listdir(group_input_dir)) if f.lower().endswith(ext_filters)]
        
        if not files:
            continue

        for f in files:
            src = os.path.join(group_input_dir, f)
            prefix = os.path.splitext(f)[0]
            
            try:
                process_and_save_png(src, out_group_dir, prefix, crop_bboxes_list)
            except Exception as e:
                print(f"  [错误] 处理 {src} 失败：{e}")

        print(f"  [完成] 组 {group_name} -> 已保存至 {out_group_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对模型输出的分类 PNG 结果生成子图切片和带框总图")
    parser.add_argument("--input_dir", required=True, help="模型结果父目录（包含多个模型子目录，如 starfm, swinstf等）")
    parser.add_argument("--output_dir", required=True, help="裁切结果保存的根目录")
    parser.add_argument("--bboxes", required=True, nargs="+",
                        help="用空格分隔的 bbox 字符串，格式 y0,y1,x0,x1。可传多个，例如: --bboxes 560,944,520,904 140,524,700,1084")
    parser.add_argument("--filter", default="", help="可选：只处理子目录名中包含该字符串的模型")
    args = parser.parse_args()

    # 解析 bbox 参数
    bbox_list = []
    for s in args.bboxes:
        parts = [int(x) for x in s.split(",")]
        if len(parts) != 4:
            raise ValueError("每个 bbox 必须为四个整数：y_start,y_end,x_start,x_end")
        bbox_list.append(parts)

    input_root = args.input_dir
    output_root = args.output_dir
    os.makedirs(output_root, exist_ok=True)

    # 遍历父目录下的所有模型子目录 (starfm, swinstf, GPSTFDiff...)
    entries = sorted(os.listdir(input_root))
    subdirs = [e for e in entries if os.path.isdir(os.path.join(input_root, e))]
    
    if len(subdirs) > 0:
        for sub in subdirs:
            if args.filter and args.filter not in sub:
                continue
            model_input_dir = os.path.join(input_root, sub)
            model_output_dir = os.path.join(output_root, sub)
            
            print(f"\n[{sub}] 开始处理...")
            crop_groups_folder(model_input_dir, model_output_dir, bbox_list)
    else:
        # 兼容单层目录的情况
        print(f"\n[直接处理] {input_root} ...")
        crop_groups_folder(input_root, output_root, bbox_list)
        
    print("\n✅ 所有模型的分类图裁切完成！")

"""
    python tools/analysis/sub_classifer_map.py \
    --input_dir /home/zhaojiazhen/workspace/STF/STF/artifacts/classfier/all_methods_eval_4 \
    --output_dir /home/zhaojiazhen/workspace/STF/STF/artifacts/classfier/all_methods_eval_4_patch \
    --bboxes 560,944,520,904 140,524,700,1084
"""
