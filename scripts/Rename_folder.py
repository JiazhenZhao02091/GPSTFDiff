import os
import re

def standardize_folder_names(root_path):
    # 正则表达式解释：
    # ^([LM])_       : 匹配以 L 或 M 开头，后跟下划线
    # (\d{4})        : 匹配 4 位数字（年份）
    # -(\d{1,2})     : 匹配横杠和 1-2 位数字（月份）
    # -(\d{1,2})$    : 匹配横杠和 1-2 位数字（日期），并以此结尾
    pattern = re.compile(r'^([LM])_(\d{4})-(\d{1,2})-(\d{1,2})$')

    # 检查路径是否存在
    if not os.path.exists(root_path):
        print(f"路径不存在: {root_path}")
        return

    count = 0
    # 遍历指定目录下的所有文件夹
    for folder_name in os.listdir(root_path):
        full_path = os.path.join(root_path, folder_name)
        
        # 只处理文件夹
        if os.path.isdir(full_path):
            match = pattern.match(folder_name)
            if match:
                prefix = match.group(1)   # L 或 M
                year = match.group(2)     # 年
                month = int(match.group(3)) # 月
                day = int(match.group(4))   # 日

                # 格式化为两位数，例如 9 变为 09
                new_name = f"{prefix}_{year}-{month:02d}-{day:02d}"

                # 如果新旧名称不同，则重命名
                if folder_name != new_name:
                    new_full_path = os.path.join(root_path, new_name)
                    
                    # 防止目标文件夹已存在导致的冲突
                    if not os.path.exists(new_full_path):
                        os.rename(full_path, new_full_path)
                        print(f"已重命名: {folder_name} -> {new_name}")
                        count += 1
                    else:
                        print(f"跳过: {new_name} 已存在")

    print(f"\n处理完成，共修改了 {count} 个文件夹。")

# --- 使用示例 ---
target_dir = "data/spatio_temporal_fusion/AHB/raw_data/Landsat/"
standardize_folder_names(target_dir)