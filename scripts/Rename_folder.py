import os
import re

def standardize_folder_names(root_path):


    pattern = re.compile(r'^([LM])_(\d{4})-(\d{1,2})-(\d{1,2})$')


    if not os.path.exists(root_path):
        print(f"路径不存在: {root_path}")
        return

    count = 0

    for folder_name in os.listdir(root_path):
        full_path = os.path.join(root_path, folder_name)


        if os.path.isdir(full_path):
            match = pattern.match(folder_name)
            if match:
                prefix = match.group(1)
                year = match.group(2)
                month = int(match.group(3))
                day = int(match.group(4))


                new_name = f"{prefix}_{year}-{month:02d}-{day:02d}"


                if folder_name != new_name:
                    new_full_path = os.path.join(root_path, new_name)


                    if not os.path.exists(new_full_path):
                        os.rename(full_path, new_full_path)
                        print(f"已重命名: {folder_name} -> {new_name}")
                        count += 1
                    else:
                        print(f"跳过: {new_name} 已存在")

    print(f"\n处理完成，共修改了 {count} 个文件夹。")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Standardize Landsat/MODIS directory names')
    parser.add_argument('directory')
    args = parser.parse_args()
    standardize_folder_names(args.directory)
