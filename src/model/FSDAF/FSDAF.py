import math
import numpy as np
import os
import datetime
import uuid
import yaml
import idlwrap
from scipy.interpolate import Rbf
import statsmodels.api as sm
from isodata import myISODATA
import argparse
import tifffile
from tqdm import tqdm

def value_locate(refx, x):
    refx = np.array(refx)
    x = np.atleast_1d(x)
    loc = np.zeros(len(x), dtype='int')
    for i in range(len(x)):
        ix = x[i]
        ind = ((refx - ix) <= 0).nonzero()[0]
        if len(ind) == 0:
            loc[i] = -1
        else:
            loc[i] = ind[-1]
    return loc

# ==================== 0. 替换 GDAL 的 I/O 辅助函数 ====================
def read_raster_tifffile(filepath):
    data = tifffile.imread(filepath)
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    return data.shape[1], data.shape[2], data

def writeimage_tifffile(data, out_name, template_path=None):
    tifffile.imwrite(out_name, data)

# ==================== 1. 批量配置与环境参数 ====================
parser = argparse.ArgumentParser(description="Run FSDAF")
parser.add_argument('--base_dir', type=str, required=True, help="输入数据目录")
parser.add_argument('--result_dir', type=str, required=True, help="输出结果目录")
args = parser.parse_args()

base_dir = args.base_dir
result_dir = args.result_dir

f = open('parameters_fsdaf.yaml')
param = yaml.safe_load(f)
w = param['w']  
num_similar_pixel = param['num_similar_pixel']  
min_class = param['min_class']  
max_class = param['max_class']
num_pure = param['num_pure']  
DN_min = param['DN_min']  
DN_max = param['DN_max']
scale_factor = 16  
block_size = param['block_size']  
background = param['background']  
background_band = param['background_band']  
I = param['I']  
maxStdv = param['maxStdv']  
minDis = param['minDis']   
minS = param['minS']   
M = param['M']  

import glob
landsat1_dir = os.path.join(base_dir, 'Landsat_01')
tif_files = glob.glob(os.path.join(landsat1_dir, '*.tif'))

print(f"共找到 {len(tif_files)} 个文件需要处理。")

for path1 in tif_files:
    img_name_landsat = os.path.basename(path1)
    img_name_modis = img_name_landsat.replace('_L_', '_M_')
    
    path2 = os.path.join(base_dir, 'MODIS_01', img_name_modis)
    path3 = os.path.join(base_dir, 'MODIS_02', img_name_modis)
    
    session_id = uuid.uuid4().hex[:6]
    base_name = img_name_landsat.replace('.tif', '')
    temp_file = os.path.join(result_dir, 'temp', f"{base_name}_{session_id}")
    if not os.path.exists(temp_file):
        os.makedirs(temp_file)
        
    out_save_dir = os.path.join(result_dir, 'Prediction_L2')
    if not os.path.exists(out_save_dir):
        os.makedirs(out_save_dir)

    fine1_whole = tifffile.imread(path1)
    if fine1_whole.ndim == 3:
        nl, ns, nb = fine1_whole.shape[0], fine1_whole.shape[1], fine1_whole.shape[2]
    else:
        nl, ns = fine1_whole.shape
        nb = 1

    suffix = os.path.splitext(path1)[-1] 

    is_hwc = False
    if fine1_whole.shape[-1] == nb:  
        fine1_whole = np.transpose(fine1_whole, (2, 0, 1))
        is_hwc = True
        nl, ns = fine1_whole.shape[1], fine1_whole.shape[2]

    orig_ns, orig_nl = ns, nl

    is_01_range = False
    if np.max(fine1_whole) <= 2.0:  
        fine1_whole = fine1_whole * 10000.0
        is_01_range = True

    patch_long = block_size * scale_factor

    n_nl = math.ceil(orig_nl / patch_long)
    n_ns = math.ceil(orig_ns / patch_long)

    ind_patch1 = np.zeros((n_nl * n_ns, 4), dtype=int)
    ind_patch = np.zeros((n_nl * n_ns, 4), dtype=int)
    location = np.zeros((n_nl * n_ns, 4), dtype=int)

    for i_ns in range(0, n_ns):
        for i_nl in range(0, n_nl):
            ind_patch1[n_ns * i_nl + i_ns, 0] = i_ns * patch_long
            ind_patch[n_ns * i_nl + i_ns, 0] = np.max([0, ind_patch1[n_ns * i_nl + i_ns, 0] - scale_factor])
            location[n_ns * i_nl + i_ns, 0] = ind_patch1[n_ns * i_nl + i_ns, 0] - ind_patch[n_ns * i_nl + i_ns, 0]

            ind_patch1[n_ns * i_nl + i_ns, 1] = np.min([ns - 1, (i_ns + 1) * patch_long - 1])
            ind_patch[n_ns * i_nl + i_ns, 1] = np.min([ns - 1, ind_patch1[n_ns * i_nl + i_ns, 1] + scale_factor])
            location[n_ns * i_nl + i_ns, 1] = ind_patch1[n_ns * i_nl + i_ns, 1] - ind_patch1[n_ns * i_nl + i_ns, 0] + location[n_ns * i_nl + i_ns, 0]

            ind_patch1[n_ns * i_nl + i_ns, 2] = i_nl * patch_long
            ind_patch[n_ns * i_nl + i_ns, 2] = np.max([0, ind_patch1[n_ns * i_nl + i_ns, 2] - scale_factor])
            location[n_ns * i_nl + i_ns, 2] = ind_patch1[n_ns * i_nl + i_ns, 2] - ind_patch[n_ns * i_nl + i_ns, 2]

            ind_patch1[n_ns * i_nl + i_ns, 3] = np.min([nl - 1, (i_nl + 1) * patch_long - 1])
            ind_patch[n_ns * i_nl + i_ns, 3] = np.min([nl - 1, ind_patch1[n_ns * i_nl + i_ns, 3] + scale_factor])
            location[n_ns * i_nl + i_ns, 3] = ind_patch1[n_ns * i_nl + i_ns, 3] - ind_patch1[n_ns * i_nl + i_ns, 2] + location[n_ns * i_nl + i_ns, 2]

    tempoutname = os.path.join(temp_file, 'temp_F1')
    for isub in range(0, n_nl * n_ns):
        col1 = ind_patch[isub, 0]; col2 = ind_patch[isub, 1]
        row1 = ind_patch[isub, 2]; row2 = ind_patch[isub, 3]
        data = fine1_whole[:, row1:row2 + 1, col1:col2 + 1]
        out_name = tempoutname + str(isub + 1) + suffix
        writeimage_tifffile(data, out_name)

    FileName2 = tifffile.imread(path2)
    if FileName2.ndim == 2:
        FileName2 = FileName2[np.newaxis, ...]
    elif is_hwc:
        FileName2 = np.transpose(FileName2, (2, 0, 1))
        
    if is_01_range:
        FileName2 = FileName2 * 10000.0
        
    tempoutname = os.path.join(temp_file, 'temp_C1')
    for isub in range(0, n_nl * n_ns):
        col1 = ind_patch[isub, 0]; col2 = ind_patch[isub, 1]
        row1 = ind_patch[isub, 2]; row2 = ind_patch[isub, 3]
        data = FileName2[:, row1:row2 + 1, col1:col2 + 1]
        out_name = tempoutname + str(isub + 1) + suffix
        writeimage_tifffile(data, out_name)

    FileName3 = tifffile.imread(path3)
    if FileName3.ndim == 2:
        FileName3 = FileName3[np.newaxis, ...]
    elif is_hwc:
        FileName3 = np.transpose(FileName3, (2, 0, 1))

    if is_01_range:
        FileName3 = FileName3 * 10000.0

    tempoutname = os.path.join(temp_file, 'temp_C0')
    for isub in range(0, n_nl * n_ns):
        col1 = ind_patch[isub, 0]; col2 = ind_patch[isub, 1]
        row1 = ind_patch[isub, 2]; row2 = ind_patch[isub, 3]
        data = FileName3[:, row1:row2 + 1, col1:col2 + 1]
        out_name = tempoutname + str(isub + 1) + suffix
        writeimage_tifffile(data, out_name)

    # (多余的强行下采样逻辑已在此处被完全删除)

    background_whole = np.zeros((nl, ns)).astype(bytes)
    ind_back = np.where(fine1_whole[background_band-1, :, :] == background)
    num_back = int(int(np.size(ind_back)) / len(ind_back))
    if num_back > 0:
        background_whole[ind_back] = 1
        for iband in range(0, nb):
            temp = fine1_whole[iband, :, :]
            temp[ind_back] = np.mean(temp[np.where(background_whole == 0)])
            fine1_whole[iband, :, :] = temp

    tempoutname11 = temp_file + "/fine1_nobackground" + suffix
    writeimage_tifffile(fine1_whole, tempoutname11)

    ind_back = 0 
    temp = 0  
    fine1_whole = 0 
    background_whole = 0 

    _, _, imagei_new = read_raster_tifffile(tempoutname11)
    imagei_new = np.maximum(imagei_new, 0)

    imagei_new_hwc = np.transpose(imagei_new, (1, 2, 0))

    params = {"K": min_class, "I": I, "P": 2, "maxStdv": maxStdv, "minDis": minDis,
            "minS": minS, "M": M}
    [labels, centers] = myISODATA(imagei_new_hwc, parameters=params)   
    labels0 = labels + 1

    tempoutname = temp_file + '/class'
    for isub in range(0, n_nl * n_ns):
        col1 = ind_patch[isub, 0]
        col2 = ind_patch[isub, 1]
        row1 = ind_patch[isub, 2]
        row2 = ind_patch[isub, 3]
        data = np.array([labels0[row1:row2 + 1, col1:col2 + 1]])
        out_name = tempoutname + str(isub + 1) + suffix
        writeimage_tifffile(data, out_name)

    starttime = datetime.datetime.now()  

    print(f'there are total {n_nl*n_ns} blocks to process')

    for isub in range(0, n_nl * n_ns):
        FileName = temp_file + '/temp_F1' + str(isub + 1) + suffix
        nl, ns, fine1 = read_raster_tifffile(FileName)

        FileName = temp_file + '/temp_C1' + str(isub + 1) + suffix
        _, _, coarse1 = read_raster_tifffile(FileName)

        FileName = temp_file + '/temp_C0' + str(isub + 1) + suffix
        _, _, coarse2 = read_raster_tifffile(FileName)

        FileName = temp_file + '/class' + str(isub + 1) + suffix
        _, _, L1_class0 = read_raster_tifffile(FileName)

        num_class = int(np.max(L1_class0))
        i_new_c = 0
        L1_class = np.zeros((nl, ns)).astype(int)
        for iclass in range(0, num_class):
            ind_ic = np.logical_and(L1_class0[0] == iclass + 1, fine1[background_band - 1, :, :] != background)
            num_ic = np.sum(ind_ic)
            if num_ic > 0:
                L1_class[ind_ic] = i_new_c + 1
                i_new_c = i_new_c + 1

        num_class = np.max(L1_class)

        if num_class > 0:  
            for ib in range(0, nb):
                fine1_band = fine1[ib, :, :]
                fine1_band_1 = fine1_band.flatten()
                sortIndex = np.argsort(fine1_band_1, kind='mergesort')
                sortIndices = (idlwrap.findgen(float(ns) * nl + 1)) / (float(ns) * nl)
                Percentiles = [0.0001, 0.9999]
                dataIndices = value_locate(sortIndices, Percentiles)
                data_1_4 = fine1_band_1[sortIndex[dataIndices]]
                
                ind_small = np.logical_or(fine1[ib, :, :] <= data_1_4[0], fine1[ib, :, :] < DN_min)
                temp = fine1[ib, :, :]
                temp[ind_small] = np.min((fine1[ib, :, :])[np.logical_and(fine1[ib, :, :] > data_1_4[0], fine1[ib, :, :] >= DN_min)])
                fine1[ib, :, :] = temp
                
                ind_large = np.logical_or(fine1[ib, :, :] >= data_1_4[1], fine1[ib, :, :] > DN_max)
                temp = fine1[ib, :, :]
                temp[ind_large] = np.max((fine1[ib, :, :])[np.logical_and(fine1[ib, :, :] < data_1_4[1], fine1[ib, :, :] <= DN_max)])
                fine1[ib, :, :] = temp

            ii = 0
            ns_c = int(np.floor(ns / scale_factor))
            nl_c = int(np.floor(nl / scale_factor))
            index_f = np.zeros((nl, ns)).astype(int)
            index_c = np.zeros((nl_c, ns_c)).astype(int)
            for i in range(0, ns_c):
                for j in range(0, nl_c):
                    index_f[j * scale_factor:(j + 1) * scale_factor, i * scale_factor:(i + 1) * scale_factor] = ii
                    index_c[j, i] = ii
                    ii = ii + 1.0

            row_ind = np.zeros((nl, ns)).astype(int)
            col_ind = np.zeros((nl, ns)).astype(int)
            for i in range(0, ns): col_ind[:, i] = i
            for i in range(0, nl): row_ind[i, :] = i

            fine_c1 = np.zeros((nb, nl_c, ns_c)).astype(float)
            coarse_c1 = np.zeros((nb, nl_c, ns_c)).astype(float)
            coarse_c2 = np.zeros((nb, nl_c, ns_c)).astype(float)
            row_c = np.zeros((nl_c, ns_c)).astype(float)
            col_c = np.zeros((nl_c, ns_c)).astype(float)
            for ic in range(0, ns_c):
                for jc in range(0, nl_c):
                    ind_c = np.where(index_f == index_c[jc, ic])
                    row_c[jc, ic] = np.mean(row_ind[ind_c])
                    col_c[jc, ic] = np.mean(col_ind[ind_c])
                    for ib in range(0, nb):
                        fine_c1[ib, jc, ic] = np.mean((fine1[ib, :, :])[ind_c])
                        coarse_c1[ib, jc, ic] = np.mean((coarse1[ib, :, :])[ind_c])
                        coarse_c2[ib, jc, ic] = np.mean((coarse2[ib, :, :])[ind_c])

            Fraction1 = np.zeros((num_class, nl_c, ns_c)).astype(float)
            for ic in range(0, ns_c):
                for jc in range(0, nl_c):
                    ind_c = np.where(index_f == index_c[jc, ic])
                    num_c = int(int(np.size(ind_c)) / len(ind_c))
                    L1_class_c = L1_class[ind_c]
                    for iclass in range(0, num_class):
                        ind_ic = np.where(L1_class_c == iclass+1)
                        num_ic = int(int(np.size(ind_ic)) / len(ind_ic))
                        Fraction1[iclass, jc, ic] = num_ic / num_c

                    if np.sum(Fraction1[:, jc, ic]) <= 0.999:  
                        Fraction1[:, jc, ic] = 0

            het_index = np.zeros((nl, ns)).astype(float)
            scale_d = w

            for i in range(0, ns):
                for j in range(0, nl):
                    ai = int(np.max([0, i - scale_d]))
                    bi = int(np.min([ns - 1, i + scale_d]))
                    aj = int(np.max([0, j - scale_d]))
                    bj = int(np.min([nl - 1, j + scale_d]))
                    class_t = L1_class[j, i]
                    ind_same_class = np.where(L1_class[aj:bj+1, ai:bi+1] == class_t)
                    num_sameclass = int(int(np.size(ind_same_class)) / len(ind_same_class))
                    het_index[j, i] = float(num_sameclass) / ((bi-ai+1.0) * (bj-aj+1.0))

            c_rate = np.zeros((nb, num_class)).astype(float)
            min_allow = np.zeros(nb).astype(float)
            max_allow = np.zeros(nb).astype(float)
            for ib in range(0, nb):
                min_allow[ib] = np.min(coarse_c2[ib, :, :] - coarse_c1[ib, :, :]) - np.std(coarse_c2[ib, :, :] - coarse_c1[ib, :, :])
                max_allow[ib] = np.max(coarse_c2[ib, :, :] - coarse_c1[ib, :, :]) + np.std(coarse_c2[ib, :, :] - coarse_c1[ib, :, :])

            for ib in range(0, nb):
                x_matrix = np.zeros((num_pure * num_class, num_class)).astype(float)
                y_matrix = np.zeros((num_pure * num_class, 1)).astype(float)
                ii = 0
                for ic in range(0, num_class):
                    order_s = np.argsort((Fraction1[ic, :, :]).flatten(), kind='mergesort')
                    order = order_s[::-1]
                    ind_f = np.where(Fraction1[ic, :, :] > 0.01)      
                    num_f = int(int(np.size(ind_f)) / len(ind_f))
                    num_pure1 = np.min([num_f, num_pure])
                    change_c = (coarse_c2[ib, :, :].flatten())[order[0:num_pure1]] - (coarse_c1[ib, :, :].flatten())[order[0:num_pure1]]

                    sortIndex = np.argsort(change_c, kind='mergesort')
                    sortIndices = (idlwrap.findgen(float(num_pure1+1))) / num_pure1
                    Percentiles = [0.1, 0.9]
                    dataIndices = value_locate(sortIndices, Percentiles)
                    data_1_4 = change_c[sortIndex[dataIndices]]
                    ind_nonchange = np.logical_and(change_c >= data_1_4[0], change_c <= data_1_4[1])
                    num_nonc = np.sum(ind_nonchange)
                    if num_nonc > 0:
                        y_matrix[ii:ii+num_nonc, 0] = change_c[ind_nonchange]
                        for icc in range(0, num_class):
                            f_c = (Fraction1[icc, :, :].flatten())[order[0:num_pure1]]
                            x_matrix[ii:ii+num_nonc, icc] = f_c[ind_nonchange]
                        ii = ii + num_nonc
                x_matrix = x_matrix[0:ii, :]
                y_matrix = y_matrix[0:ii, 0]

                model = sm.OLS(y_matrix, x_matrix).fit()
                opt = model.params
                c_rate[ib, :] = opt

            L2_1 = fine1.copy()
            for ic in range(1, num_class+1):
                ind_L1_class = np.where(L1_class == ic)
                for ib in range(0, nb):
                    temp = L2_1[ib, :, :]
                    temp[ind_L1_class] = (fine1[ib, :, :])[ind_L1_class] + c_rate[ib, ic-1]

            coarse_c2_p = np.zeros((nb, nl_c, ns_c)).astype(float)
            for ic in range(0, ns_c):
                for jc in range(0, nl_c):
                    ind_c = np.where(index_f == index_c[jc, ic])
                    for ib in range(0, nb):
                        coarse_c2_p[ib, jc, ic] = np.mean((L2_1[ib, :, :])[ind_c])

            min_allow = np.zeros(nb).astype(float)
            max_allow = np.zeros(nb).astype(float)
            for ib in range(0, nb):
                min_allow0 = np.min([np.min(coarse2[ib, :, :]), np.min(L2_1[ib, :, :])])
                min_allow[ib] = np.max([min_allow0, DN_min])
                max_allow0 = np.max([np.max(coarse2[ib, :, :]), np.max(L2_1[ib, :, :])])
                max_allow[ib] = np.min([max_allow0, DN_max])

            L2_tps = np.zeros((nb, nl, ns)).astype(float)
            for ib in range(0, nb):
                rbf = Rbf(row_c.ravel(), col_c.ravel(), (coarse_c2[ib, :, :]).ravel(), function='multiquadric')
                tps = rbf(row_ind.ravel(), col_ind.ravel()).reshape([nl, ns])
                L2_tps[ib, :, :] = tps

            print(f'Block {isub+1} - finish TPS prediction')

            predict_change_c = coarse_c2_p - fine_c1     
            real_change_c = coarse_c2 - coarse_c1        
            change_R = real_change_c - predict_change_c

            change_21_c = np.zeros((nb, nl_c, ns_c)).astype(float)
            change_21 = np.zeros((nb, nl, ns)).astype(float)

            for ic in range(0, ns_c):
                for jc in range(0, nl_c):
                    ind_c = np.where(index_f == index_c[jc, ic])
                    num_ii = int(int(np.size(ind_c)) / len(ind_c))

                    for ib in range(0, nb):
                        diff_change = change_R[ib, jc, ic]
                        w_change_tps = (L2_tps[ib, :, :])[ind_c] - (L2_1[ib, :, :])[ind_c]
                        if diff_change <= 0:
                            ind_noc = np.where(w_change_tps > 0)
                            num_noc = int(int(np.size(ind_noc)) / len(ind_noc))
                            if num_noc > 0:
                                w_change_tps[ind_noc] = 0
                        else:
                            ind_noc = np.where(w_change_tps < 0)
                            num_noc = int(int(np.size(ind_noc)) / len(ind_noc))
                            if num_noc > 0:
                                w_change_tps[ind_noc] = 0

                        w_change_tps = np.abs(w_change_tps)
                        w_unform = np.zeros(num_ii).astype(float)     
                        w_unform[:] = np.abs(diff_change)

                        w_change = w_change_tps * het_index[ind_c] + w_unform*(1.0-het_index[ind_c]) + 0.000001  
                        w_change = w_change / (np.mean(w_change))  

                        ind_extrem = np.where(w_change > 10)
                        num_extrem = int(int(np.size(ind_extrem)) / len(ind_extrem))
                        if num_extrem > 0:
                            w_change[ind_extrem] = np.mean(w_change)
                        w_change = w_change / (np.mean(w_change))

                        temp = change_21[ib, :, :]
                        temp[ind_c] = w_change * diff_change
                        change_21[ib, :, :] = temp

            fine2_2 = L2_1 + change_21
            for ib in range(0, nb):
                temp = fine2_2[ib, :, :]
                ind_min = np.where(temp < min_allow[ib])
                num_min = int(int(np.size(ind_min)) / len(ind_min))
                if num_min > 0:
                    temp[ind_min] = min_allow[ib]
                ind_max = np.where(temp > max_allow[ib])
                num_max = int(int(np.size(ind_max)) / len(ind_max))
                if num_max > 0:
                    temp[ind_max] = max_allow[ib]
                fine2_2[ib, :, :] = temp

            change_21 = fine2_2 - fine1

        else:
            change_21 = fine1 - fine1

        change_21 = change_21[:, location[isub, 2]:location[isub, 3] + 1, location[isub, 0]:location[isub, 1] + 1]

        print(f'Block {isub+1} - finish change prediction')
        tempoutname1 = temp_file + '/temp_change'
        Out_Name = tempoutname1 + str(isub + 1) + suffix
        writeimage_tifffile(change_21, Out_Name)

    datalist = []
    minx_list = []
    maxX_list = []
    minY_list = []
    maxY_list = []

    for isub in range(0, n_ns * n_nl):
        out_name = temp_file + '/temp_change' + str(isub + 1) + suffix
        datalist.append(out_name)
        col1 = ind_patch1[isub, 0]; col2 = ind_patch1[isub, 1]
        row1 = ind_patch1[isub, 2]; row2 = ind_patch1[isub, 3]
        minx_list.append(col1)
        maxX_list.append(col2)
        minY_list.append(row1)
        maxY_list.append(row2)

    minX = min(minx_list)
    minY = min(minY_list)

    xOffset_list = []
    yOffset_list = []
    for i in range(len(datalist)):
        xOffset_list.append(int(minx_list[i] - minX))
        yOffset_list.append(int(minY_list[i] - minY))

    img_name_out = img_name_landsat.replace('.tif', f'_FSDAF{suffix}')
    path_change = os.path.join(out_save_dir, img_name_out)
    
    change_mosaic = np.zeros((nb, orig_nl, orig_ns), dtype=np.float32)

    for i, data_path in enumerate(datalist):
        _, _, datavalue = read_raster_tifffile(data_path)
        
        if is_01_range:
            datavalue = datavalue / 10000.0

        y_start = yOffset_list[i]
        x_start = xOffset_list[i]
        y_end = y_start + datavalue.shape[1]
        x_end = x_start + datavalue.shape[2]
        
        change_mosaic[:, y_start:y_end, x_start:x_end] = datavalue

    if is_hwc:
        save_change_mosaic = np.transpose(change_mosaic, (1, 2, 0))
    else:
        save_change_mosaic = change_mosaic

    tifffile.imwrite(path_change, save_change_mosaic)

    FileName6 = temp_file + "/temp_change" + suffix
    _, _, change = read_raster_tifffile(FileName6)

    tempoutname = temp_file + '/temp_change'
    for isub in range(0, n_nl * n_ns):
        col1 = ind_patch[isub, 0]; col2 = ind_patch[isub, 1]
        row1 = ind_patch[isub, 2]; row2 = ind_patch[isub, 3]
        data = change[:, row1:row2 + 1, col1:col2 + 1]
        out_name = tempoutname + str(isub + 1) + suffix
        writeimage_tifffile(data, out_name)

    for isub in range(0, n_nl * n_ns):
        print(f"\n=================== Starting Final Prediction for Block {isub+1}/{n_nl * n_ns} ===================")
        FileName = temp_file + '/temp_F1' + str(isub + 1) + suffix
        nl, ns, fine1 = read_raster_tifffile(FileName)

        FileName = temp_file + '/temp_C1' + str(isub + 1) + suffix
        _, _, coarse1 = read_raster_tifffile(FileName)

        FileName = temp_file + '/temp_C0' + str(isub + 1) + suffix
        _, _, coarse2 = read_raster_tifffile(FileName)

        FileName = temp_file + '/class' + str(isub + 1) + suffix
        _, _, L1_class = read_raster_tifffile(FileName)

        FileName = temp_file + '/temp_change' + str(isub + 1) + suffix
        _, _, change_21 = read_raster_tifffile(FileName)

        fine2 = np.zeros([nb, location[isub, 3]-location[isub, 2]+1, location[isub, 1]-location[isub, 0]+1]).astype(float)

        D_temp1 = w - np.tile((idlwrap.indgen(w*2+1)), (int(w*2+1), 1))
        d1 = np.power(D_temp1, 2)
        D_temp2 = w - np.tile(idlwrap.indgen(1, w*2+1), (1, int(w*2+1)))
        d2 = np.power(D_temp2, 2)
        D_D_all = np.sqrt(d1 + d2)
        D_D_all = D_D_all.flatten()

        similar_th = np.zeros(nb).astype(float)
        for iband in range(0, nb):
            similar_th[iband] = np.std(fine1[iband, :, :]) * 2.0 / float(num_class)

        for i in tqdm(range(location[isub, 0], location[isub, 1] + 1), desc=f"Block {isub+1} Pixel Process"):
            for j in range(location[isub, 2], location[isub, 3] + 1):
                if fine1[background_band - 1, j, i] != background:     

                    ai = int(np.max([0, i - w]))
                    bi = int(np.min([ns - 1, i + w]))
                    aj = int(np.max([0, j - w]))
                    bj = int(np.min([nl - 1, j + w]))

                    ci = i - ai   
                    cj = j - aj

                    col_wind = np.tile(idlwrap.indgen(bi-ai+1), (int(bj-aj+1), 1))
                    row_wind = np.tile(idlwrap.indgen(1, bj-aj+1), (1, int(bi-ai+1)))

                    similar_cand = np.zeros((bi-ai+1)*(bj-aj+1)).astype(float)      
                    position_cand = np.zeros((bi-ai+1)*(bj-aj+1)).astype(int) + 1   
                    for ib in range(0, nb):
                        cand_band = np.zeros((bi-ai+1)*(bj-aj+1)).astype(int)
                        wind_fine = fine1[ib, aj:bj+1, ai:bi+1]
                        S_S = np.abs(wind_fine - wind_fine[cj, ci])
                        similar_cand = similar_cand + (S_S / (wind_fine[cj, ci] + 0.00000001)).flatten()
                        ind_cand = np.where(S_S.flatten() < similar_th[ib])
                        cand_band[ind_cand] = 1
                        position_cand = position_cand * cand_band

                    indcand = np.where(position_cand != 0)
                    number_cand0 = int(int(np.size(indcand)) / len(indcand))   
                    if (bi-ai+1) * (bj-aj+1) < (w*2.0+1) * (w*2.0+1):   
                        distance_cand = np.sqrt((ci-col_wind)**2 + (cj-row_wind)**2) + 0.00001
                    else:
                        distance_cand = D_D_all    

                    combine_similar_cand = (similar_cand+0.00001)*(10.0+distance_cand/w).flatten()          
                    order_dis = np.argsort(combine_similar_cand[indcand], kind='mergesort')
                    number_cand = np.min([number_cand0, num_similar_pixel])
                    ind_same_class = (indcand[0])[order_dis[0:int(number_cand)]]           

                    D_D_cand = (distance_cand.flatten())[ind_same_class]
                    C_D = (1.0+D_D_cand/w) * (similar_cand[ind_same_class]+1.0)
                    C_D = 1.0 / C_D
                    weight = C_D / np.sum(C_D)

                    for iband in range(0, nb):
                        change_21_win = change_21[iband, aj:bj+1, ai:bi+1]
                        change_cand = (change_21_win.flatten())[ind_same_class]
                        fine2[iband, j-location[isub, 2], i - location[isub, 0]] = fine1[iband, j, i] + np.sum(weight * change_cand)

                        if fine2[iband, j-location[isub, 2], i - location[isub, 0]] < DN_min:
                            another_predict = np.max([DN_min, fine1[iband, j, i]] + coarse2[iband, j, i] - coarse1[iband, j, i])
                            fine2[iband, j-location[isub, 2], i - location[isub, 0]] = np.min([DN_max, another_predict])

                        if fine2[iband, j-location[isub, 2], i - location[isub, 0]] > DN_max:
                            another_predict = np.min([DN_max, fine1[iband, j, i]] + coarse2[iband, j, i] - coarse1[iband, j, i])
                            fine2[iband, j - location[isub, 2], i - location[isub, 0]] = np.max([DN_min, another_predict])

        tempoutname1 = temp_file + '/temp_blended'
        Out_Name = tempoutname1 + str(isub+1) + suffix
        writeimage_tifffile(fine2, Out_Name)

    endtime = datetime.datetime.now()
    print(f'\nTotal Time Used: {(endtime - starttime).seconds} seconds')

    datalist = []
    minx_list = []
    minY_list = []

    for isub in range(0, n_ns * n_nl):
        out_name = temp_file + '/temp_blended' + str(isub+1) + suffix
        datalist.append(out_name)
        col1 = ind_patch1[isub, 0]
        row1 = ind_patch1[isub, 2]
        minx_list.append(col1)
        minY_list.append(row1)

    minX = min(minx_list)
    minY = min(minY_list)

    xOffset_list = []
    yOffset_list = []
    for i in range(len(datalist)):
        xOffset_list.append(int(minx_list[i] - minX))
        yOffset_list.append(int(minY_list[i] - minY))

    path_final = os.path.splitext(path3)[0] + "_FSDAF" + suffix
    final_mosaic = np.zeros((nb, orig_nl, orig_ns), dtype=np.float32)

    for i, data_path in enumerate(datalist):
        _, _, datavalue = read_raster_tifffile(data_path)
        
        if is_01_range:
            datavalue = datavalue / 10000.0

        y_start = yOffset_list[i]
        x_start = xOffset_list[i]
        y_end = y_start + datavalue.shape[1]
        x_end = x_start + datavalue.shape[2]
        
        final_mosaic[:, y_start:y_end, x_start:x_end] = datavalue

    if is_hwc:
        save_final_mosaic = np.transpose(final_mosaic, (1, 2, 0))
    else:
        save_final_mosaic = final_mosaic

    tifffile.imwrite(path_final, save_final_mosaic)