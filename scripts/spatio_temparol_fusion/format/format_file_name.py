def format_file_name(src_name):
    src_stem = src_name.split('.')[0]
    suffix = src_name.split('.')[-1]
    if 'TM' in src_stem:
        date = src_stem.split('_')[0]
        year = date[:4]
        month = date[4:6]
        day = date[-2:]
        tar_name = 'L_{}-{}-{}.{}'.format(year, month, day, suffix)
    elif 'MODIS_McLean' in src_stem:
        parts = src_stem.split('_')
        year = parts[-3]
        month = parts[-2]
        day = parts[-1]
        tar_name = 'M_{}-{}-{}.{}'.format(year, month, day, suffix)
    elif 'MOD' in src_stem:
        date = src_stem.split('_')[-1]
        year = date[:4]
        month = date[4:6]
        day = date[-2:]
        tar_name = 'M_{}-{}-{}.{}'.format(year, month, day, suffix)
    elif 'Landsat_8' in src_stem:
        parts = src_stem.split('_')
        year = parts[-3]
        month = parts[-2]
        day = parts[-1]
        tar_name = 'L_{}-{}-{}.{}'.format(year, month, day, suffix)
    else:
        tar_name = src_name[0] + '_' + src_name[2:]
    return tar_name

if __name__ == '__main__':
    source_name_MODIS = "L_2021-03-02.tif"
    source_name_Landsat = "M_2021-03-02.tif"
    formatted_name_MODIS = format_file_name(source_name_MODIS)
    formatted_name_Landsat = format_file_name(source_name_Landsat)
    print(formatted_name_MODIS)
    print(formatted_name_Landsat)
    pass
