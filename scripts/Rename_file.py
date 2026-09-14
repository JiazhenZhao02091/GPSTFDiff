import os
import re

def rename_files(directory):
    # Regex pattern to match filenames like L_2016-8-26.tif or M_2016-8-1.tif
    # Group 1: Prefix (L or M)
    # Group 2: Year
    # Group 3: Month
    # Group 4: Day
    # Group 5: Extension
    pattern = re.compile(r"^([LM])-(\d{4})-(\d{1,2})-(\d{1,2})(\.tif)$")

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            prefix, year, month, day, ext = match.groups()

            # Zero-pad month and day to ensure they are 2 digits
            new_month = month.zfill(2)
            new_day = day.zfill(2)

            new_filename = f"{prefix}_{year}-{new_month}-{new_day}{ext}"
            
            # Skip if the filename is already in the correct format
            if filename != new_filename:
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_filename)
                
                print(f"Renaming: {filename} -> {new_filename}")
                os.rename(old_path, new_path)

if __name__ == "__main__":
    # Change this path to the directory containing your images
    target_directory = "/home/zhaojiazhen/workspace/STF/STF/data/spatio_temporal_fusion/Tianjin/raw_data/MODIS" 
    rename_files(target_directory)