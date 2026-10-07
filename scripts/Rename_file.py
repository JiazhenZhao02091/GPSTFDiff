import os
import re

def rename_files(directory):


    pattern = re.compile(r"^([LM])-(\d{4})-(\d{1,2})-(\d{1,2})(\.tif)$")

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            prefix, year, month, day, ext = match.groups()


            new_month = month.zfill(2)
            new_day = day.zfill(2)

            new_filename = f"{prefix}_{year}-{new_month}-{new_day}{ext}"


            if filename != new_filename:
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_filename)

                print(f"Renaming: {filename} -> {new_filename}")
                os.rename(old_path, new_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Standardize Landsat/MODIS TIFF filenames')
    parser.add_argument('directory')
    args = parser.parse_args()
    rename_files(args.directory)
