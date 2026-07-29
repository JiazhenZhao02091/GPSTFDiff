import math
import numpy as np
import os
import glob
import datetime
import tifffile
from scipy.interpolate import Rbf
import statsmodels.api as sm
from isodata import myISODATA

def value_locate(refx, x):
    refx = np.array(refx)
    x = np.atleast_1d(x)
    loc = np.zeros(len(x), dtype=int)
    for i in range(len(x)):
        ix = x[i]
        ind = ((refx - ix) <= 0).nonzero()[0]
        if len(ind) == 0:
            loc[i] = -1
        else:
            loc[i] = ind[-1]
    return loc

def read_raster(infile):
    """使用 tifffile 读取栅格数据，智能处理 GDAL 导出的多页 TIFF，并转换为 (bands, rows, cols)"""
    with tifffile.TiffFile(infile) as tif:
        # 很多通过 GDAL/MATLAB 导出的多波段 TIFF 会把波段存在不同的 page (IFD) 里
        if len(tif.pages) > 1:
            try:
                # 强制遍历所有页面并堆叠，确保 6 个波段被完整读出
                data = np.stack([page.asarray() for page in tif.pages])
            except Exception:
                data = tif.asarray()
        else:
            data = tif.asarray()
            
    orig_dtype = data.dtype
    data = data.astype(np.float32)
    data = np.squeeze(data)
    
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    elif data.ndim == 3:
        min_dim_idx = np.argmin(data.shape)
        if min_dim_idx != 0 and data.shape[min_dim_idx] <= 30:
            data = np.moveaxis(data, min_dim_idx, 0)
    elif data.ndim > 3:
        data = data.reshape(-1, data.shape[-2], data.shape[-1])
        
    rows, cols = data.shape[-2], data.shape[-1]
    return rows, cols, data, orig_dtype

def writeimage(bands, path, out_dtype=np.float32):
    """使用 tifffile 写入栅格数据，按输入原生类型保存"""
    if bands is None or len(bands) == 0:
        return
    bands_out = np.array(bands, dtype=out_dtype)
    tifffile.imwrite(path, bands_out)

def run_fsdaf_single_pair(path_l1, path_m1, path_m2, out_path, value_range, params):
    w = params.get('w', 20)
    num_similar_pixel = params.get('num_similar_pixel', 20)
    min_class = params.get('min_class', 4.0)
    max_class = params.get('max_class', 6.0)
    num_pure = params.get('num_pure', 100)
    DN_min = float(value_range[0])
    DN_max = float(value_range[1])
    scale_factor = params.get('scale_factor', 16)
    block_size = params.get('block_size', 30)
    background = params.get('background', -9999)
    background_band = params.get('background_band', 3)
    I = params.get('I', 20)
    maxStdv = params.get('maxStdv', 500)
    minDis = params.get('minDis', 500)
    minS = params.get('minS', 200)
    M_param = params.get('M', 0.05)

    print(f"Loading data...")
    orig_nl, orig_ns, fine1_whole, orig_dtype = read_raster(path_l1)
    _, _, coarse1_whole, _ = read_raster(path_m1)
    _, _, coarse2_whole, _ = read_raster(path_m2)
    
    nb = fine1_whole.shape[0]
    min_b = min(nb, coarse1_whole.shape[0], coarse2_whole.shape[0])
    if nb != min_b or coarse1_whole.shape[0] != min_b or coarse2_whole.shape[0] != min_b:
        print(f"   [Warning] 波段数量不一致！自动截取至 {min_b} 个波段。")
        fine1_whole = fine1_whole[:min_b]
        coarse1_whole = coarse1_whole[:min_b]
        coarse2_whole = coarse2_whole[:min_b]
        nb = min_b

    bg_idx = min(background_band - 1, nb - 1)

    background_whole = np.zeros((orig_nl, orig_ns), dtype=int)
    ind_back = np.where(fine1_whole[bg_idx, :, :] == background)
    num_back = ind_back[0].size
    
    if num_back > 0:
        background_whole[ind_back] = 1
        for iband in range(nb):
            temp = fine1_whole[iband, :, :].copy()
            temp[ind_back] = np.mean(temp[np.where(background_whole == 0)])
            fine1_whole[iband, :, :] = temp

    print("Classifying fine image at T1...")
    imagei_new = np.transpose(fine1_whole, (1, 2, 0)) 
    imagei_new = np.maximum(imagei_new, 0)
    
    isodata_params = {"K": int(min_class), "I": int(I), "P": 2, "maxStdv": maxStdv, "minDis": minDis, "minS": int(minS), "M": M_param}
    labels, centers = myISODATA(imagei_new, parameters=isodata_params)
    L1_class_whole = labels + 1

    patch_long = block_size * scale_factor
    n_nl = math.ceil(orig_nl / patch_long)
    n_ns = math.ceil(orig_ns / patch_long)

    ind_patch1 = np.zeros((n_nl * n_ns, 4), dtype=int)
    ind_patch = np.zeros((n_nl * n_ns, 4), dtype=int)
    location = np.zeros((n_nl * n_ns, 4), dtype=int)

    for i_ns in range(n_ns):
        for i_nl in range(n_nl):
            idx = n_ns * i_nl + i_ns
            ind_patch1[idx, 0] = i_ns * patch_long
            ind_patch[idx, 0] = np.max([0, ind_patch1[idx, 0] - scale_factor])
            location[idx, 0] = ind_patch1[idx, 0] - ind_patch[idx, 0]

            ind_patch1[idx, 1] = np.min([orig_ns - 1, (i_ns + 1) * patch_long - 1])
            ind_patch[idx, 1] = np.min([orig_ns - 1, ind_patch1[idx, 1] + scale_factor])
            location[idx, 1] = ind_patch1[idx, 1] - ind_patch1[idx, 0] + location[idx, 0]

            ind_patch1[idx, 2] = i_nl * patch_long
            ind_patch[idx, 2] = np.max([0, ind_patch1[idx, 2] - scale_factor])
            location[idx, 2] = ind_patch1[idx, 2] - ind_patch[idx, 2]

            ind_patch1[idx, 3] = np.min([orig_nl - 1, (i_nl + 1) * patch_long - 1])
            ind_patch[idx, 3] = np.min([orig_nl - 1, ind_patch1[idx, 3] + scale_factor])
            location[idx, 3] = ind_patch1[idx, 3] - ind_patch1[idx, 2] + location[idx, 2]

    starttime = datetime.datetime.now()
    print(f'Total blocks to process: {n_nl * n_ns}')

    change_full = np.zeros((nb, orig_nl, orig_ns), dtype=np.float32)
    fine2_full = np.zeros((nb, orig_nl, orig_ns), dtype=np.float32)

    # ========================== PASS 1 ==========================
    for isub in range(n_nl * n_ns):
        c1, c2 = ind_patch[isub, 0], ind_patch[isub, 1]
        r1, r2 = ind_patch[isub, 2], ind_patch[isub, 3]

        fine1 = fine1_whole[:, r1:r2 + 1, c1:c2 + 1].copy()
        coarse1 = coarse1_whole[:, r1:r2 + 1, c1:c2 + 1].copy()
        coarse2 = coarse2_whole[:, r1:r2 + 1, c1:c2 + 1].copy()
        L1_class0 = L1_class_whole[r1:r2 + 1, c1:c2 + 1].copy()

        nl, ns = fine1.shape[1], fine1.shape[2]

        num_class = int(np.max(L1_class0))
        i_new_c = 0
        L1_class = np.zeros((nl, ns), dtype=int)
        for iclass in range(num_class):
            ind_ic = np.logical_and(L1_class0 == iclass + 1, fine1[bg_idx, :, :] != background)
            if ind_ic.sum() > 0:
                L1_class[ind_ic] = i_new_c + 1
                i_new_c += 1
        num_class = np.max(L1_class)

        if num_class > 0:
            for ib in range(nb):
                fine1_band = fine1[ib, :, :]
                fine1_band_1 = fine1_band.flatten()
                sortIndex = np.argsort(fine1_band_1, kind='mergesort')
                sortIndices = np.linspace(0, 1, ns * nl)
                Percentiles = [0.0001, 0.9999]
                dataIndices = value_locate(sortIndices, Percentiles)
                data_1_4 = fine1_band_1[sortIndex[dataIndices]]
                
                ind_small = np.logical_or(fine1[ib, :, :] <= data_1_4[0], fine1[ib, :, :] < DN_min)
                temp = fine1[ib, :, :]
                valid_min = temp[np.logical_and(temp > data_1_4[0], temp >= DN_min)]
                if valid_min.size > 0: temp[ind_small] = np.min(valid_min)
                
                ind_large = np.logical_or(fine1[ib, :, :] >= data_1_4[1], fine1[ib, :, :] > DN_max)
                valid_max = temp[np.logical_and(temp < data_1_4[1], temp <= DN_max)]
                if valid_max.size > 0: temp[ind_large] = np.max(valid_max)
                fine1[ib, :, :] = temp

            ns_c, nl_c = int(np.floor(ns / scale_factor)), int(np.floor(nl / scale_factor))
            index_f, index_c = np.zeros((nl, ns), dtype=int), np.zeros((nl_c, ns_c), dtype=int)
            ii = 0
            for i in range(ns_c):
                for j in range(nl_c):
                    index_f[j * scale_factor:(j + 1) * scale_factor, i * scale_factor:(i + 1) * scale_factor] = ii
                    index_c[j, i] = ii
                    ii += 1

            col_ind, row_ind = np.meshgrid(np.arange(ns), np.arange(nl))
            fine_c1, coarse_c1, coarse_c2 = np.zeros((nb, nl_c, ns_c)), np.zeros((nb, nl_c, ns_c)), np.zeros((nb, nl_c, ns_c))
            row_c, col_c = np.zeros((nl_c, ns_c)), np.zeros((nl_c, ns_c))

            for ic in range(ns_c):
                for jc in range(nl_c):
                    ind_c = np.where(index_f == index_c[jc, ic])
                    row_c[jc, ic], col_c[jc, ic] = np.mean(row_ind[ind_c]), np.mean(col_ind[ind_c])
                    for ib in range(nb):
                        fine_c1[ib, jc, ic] = np.mean(fine1[ib, :, :][ind_c])
                        coarse_c1[ib, jc, ic] = np.mean(coarse1[ib, :, :][ind_c])
                        coarse_c2[ib, jc, ic] = np.mean(coarse2[ib, :, :][ind_c])

            Fraction1 = np.zeros((num_class, nl_c, ns_c))
            for ic in range(ns_c):
                for jc in range(nl_c):
                    ind_c = np.where(index_f == index_c[jc, ic])
                    num_c = ind_c[0].size
                    L1_class_c = L1_class[ind_c]
                    for iclass in range(num_class):
                        num_ic = np.where(L1_class_c == iclass + 1)[0].size
                        Fraction1[iclass, jc, ic] = num_ic / num_c
                    if np.sum(Fraction1[:, jc, ic]) <= 0.999:
                        Fraction1[:, jc, ic] = 0

            het_index = np.zeros((nl, ns))
            for i in range(ns):
                for j in range(nl):
                    ai, bi = max(0, i - w), min(ns - 1, i + w)
                    aj, bj = max(0, j - w), min(nl - 1, j + w)
                    class_t = L1_class[j, i]
                    num_sameclass = np.where(L1_class[aj:bj + 1, ai:bi + 1] == class_t)[0].size
                    het_index[j, i] = float(num_sameclass) / ((bi - ai + 1.0) * (bj - aj + 1.0))

            c_rate = np.zeros((nb, num_class))
            for ib in range(nb):
                x_matrix, y_matrix = np.zeros((num_pure * num_class, num_class)), np.zeros((num_pure * num_class, 1))
                ii_idx = 0
                for ic in range(num_class):
                    order = np.argsort(Fraction1[ic, :, :].flatten(), kind='mergesort')[::-1]
                    num_f = np.where(Fraction1[ic, :, :] > 0.01)[0].size
                    num_pure1 = min(num_f, num_pure)
                    if num_pure1 == 0: continue
                    
                    change_c = (coarse_c2[ib, :, :].flatten())[order[0:num_pure1]] - (coarse_c1[ib, :, :].flatten())[order[0:num_pure1]]
                    sortIndex = np.argsort(change_c, kind='mergesort')
                    dataIndices = value_locate(np.linspace(0, 1, num_pure1), [0.1, 0.9])
                    data_1_4 = change_c[sortIndex[dataIndices]]
                    
                    ind_nonchange = np.logical_and(change_c >= data_1_4[0], change_c <= data_1_4[1])
                    num_nonc = ind_nonchange.sum()
                    if num_nonc > 0:
                        y_matrix[ii_idx:ii_idx + num_nonc, 0] = change_c[ind_nonchange]
                        for icc in range(num_class):
                            f_c = (Fraction1[icc, :, :].flatten())[order[0:num_pure1]]
                            x_matrix[ii_idx:ii_idx + num_nonc, icc] = f_c[ind_nonchange]
                        ii_idx += num_nonc
                
                if ii_idx > 0:
                    model = sm.OLS(y_matrix[0:ii_idx, 0], x_matrix[0:ii_idx, :]).fit()
                    c_rate[ib, :] = model.params

            L2_1 = fine1.copy()
            for ic in range(1, num_class + 1):
                ind_L1_class = np.where(L1_class == ic)
                for ib in range(nb):
                    L2_1[ib, :, :][ind_L1_class] = fine1[ib, :, :][ind_L1_class] + c_rate[ib, ic - 1]

            coarse_c2_p = np.zeros((nb, nl_c, ns_c))
            for ic in range(ns_c):
                for jc in range(nl_c):
                    ind_c = np.where(index_f == index_c[jc, ic])
                    for ib in range(nb):
                        coarse_c2_p[ib, jc, ic] = np.mean(L2_1[ib, :, :][ind_c])

            L2_tps = np.zeros((nb, nl, ns))
            for ib in range(nb):
                rbf = Rbf(row_c.ravel(), col_c.ravel(), coarse_c2[ib, :, :].ravel(), function='multiquadric')
                L2_tps[ib, :, :] = rbf(row_ind.ravel(), col_ind.ravel()).reshape(nl, ns)

            predict_change_c = coarse_c2_p - fine_c1
            real_change_c = coarse_c2 - coarse_c1
            change_R = real_change_c - predict_change_c

            change_21 = np.zeros((nb, nl, ns))
            for ic in range(ns_c):
                for jc in range(nl_c):
                    ind_c = np.where(index_f == index_c[jc, ic])
                    num_ii = ind_c[0].size
                    for ib in range(nb):
                        diff_change = change_R[ib, jc, ic]
                        w_change_tps = L2_tps[ib, :, :][ind_c] - L2_1[ib, :, :][ind_c]
                        
                        if diff_change <= 0:
                            w_change_tps[w_change_tps > 0] = 0
                        else:
                            w_change_tps[w_change_tps < 0] = 0

                        w_change_tps = np.abs(w_change_tps)
                        w_unform = np.full(num_ii, np.abs(diff_change))
                        
                        w_change = w_change_tps * het_index[ind_c] + w_unform * (1.0 - het_index[ind_c]) + 1e-6
                        w_change /= np.mean(w_change)
                        w_change[w_change > 10] = np.mean(w_change)
                        w_change /= np.mean(w_change)

                        change_21[ib, :, :][ind_c] = w_change * diff_change

            fine2_2 = L2_1 + change_21
            for ib in range(nb):
                min_allow = max(min(np.min(coarse2[ib, :, :]), np.min(L2_1[ib, :, :])), DN_min)
                max_allow = min(max(np.max(coarse2[ib, :, :]), np.max(L2_1[ib, :, :])), DN_max)
                fine2_2[ib, :, :][fine2_2[ib, :, :] < min_allow] = min_allow
                fine2_2[ib, :, :][fine2_2[ib, :, :] > max_allow] = max_allow

            change_21 = fine2_2 - fine1
        else:
            change_21 = np.zeros_like(fine1)

        loc_r1, loc_r2 = location[isub, 2], location[isub, 3]
        loc_c1, loc_c2 = location[isub, 0], location[isub, 1]
        R1, R2 = ind_patch1[isub, 2], ind_patch1[isub, 3]
        C1, C2 = ind_patch1[isub, 0], ind_patch1[isub, 1]
        
        change_full[:, R1:R2 + 1, C1:C2 + 1] = change_21[:, loc_r1:loc_r2 + 1, loc_c1:loc_c2 + 1]

    # ========================== PASS 2 ==========================
    for isub in range(n_nl * n_ns):
        c1, c2 = ind_patch[isub, 0], ind_patch[isub, 1]
        r1, r2 = ind_patch[isub, 2], ind_patch[isub, 3]

        fine1 = fine1_whole[:, r1:r2 + 1, c1:c2 + 1]
        coarse1 = coarse1_whole[:, r1:r2 + 1, c1:c2 + 1]
        coarse2 = coarse2_whole[:, r1:r2 + 1, c1:c2 + 1]
        change_21 = change_full[:, r1:r2 + 1, c1:c2 + 1]
        
        nl, ns = fine1.shape[1], fine1.shape[2]
        fine2_sub = np.zeros([nb, location[isub, 3] - location[isub, 2] + 1, location[isub, 1] - location[isub, 0] + 1])

        col_grid, row_grid = np.meshgrid(np.arange(w * 2 + 1), np.arange(w * 2 + 1))
        D_D_all = np.sqrt((w - col_grid)**2 + (w - row_grid)**2).flatten()
        
        similar_th = np.array([np.std(fine1[ib, :, :]) * 2.0 / num_class for ib in range(nb)])

        for i in range(location[isub, 0], location[isub, 1] + 1):
            for j in range(location[isub, 2], location[isub, 3] + 1):
                if fine1[bg_idx, j, i] != background:
                    ai, bi = max(0, i - w), min(ns - 1, i + w)
                    aj, bj = max(0, j - w), min(nl - 1, j + w)
                    ci, cj = i - ai, j - aj

                    col_wind, row_wind = np.meshgrid(np.arange(bi - ai + 1), np.arange(bj - aj + 1))

                    similar_cand = np.zeros((bi - ai + 1) * (bj - aj + 1))
                    position_cand = np.ones((bi - ai + 1) * (bj - aj + 1), dtype=int)
                    
                    for ib in range(nb):
                        wind_fine = fine1[ib, aj:bj + 1, ai:bi + 1]
                        S_S = np.abs(wind_fine - wind_fine[cj, ci])
                        similar_cand += (S_S / (wind_fine[cj, ci] + 1e-8)).flatten()
                        position_cand *= (S_S.flatten() < similar_th[ib]).astype(int)

                    indcand = np.where(position_cand != 0)[0]
                    number_cand0 = indcand.size
                    
                    if (bi - ai + 1) * (bj - aj + 1) < (w * 2.0 + 1) ** 2:
                        distance_cand = np.sqrt((ci - col_wind)**2 + (cj - row_wind)**2).flatten() + 1e-5
                    else:
                        distance_cand = D_D_all

                    combine_similar_cand = (similar_cand + 1e-5) * (10.0 + distance_cand / w)
                    order_dis = np.argsort(combine_similar_cand[indcand], kind='mergesort')
                    number_cand = min(number_cand0, num_similar_pixel)
                    ind_same_class = indcand[order_dis[:int(number_cand)]]

                    D_D_cand = distance_cand[ind_same_class]
                    C_D = 1.0 / ((1.0 + D_D_cand / w) * (similar_cand[ind_same_class] + 1.0))
                    weight = C_D / np.sum(C_D)

                    for ib in range(nb):
                        change_cand = change_21[ib, aj:bj + 1, ai:bi + 1].flatten()[ind_same_class]
                        pred_val = fine1[ib, j, i] + np.sum(weight * change_cand)
                        
                        min_bound = DN_min
                        max_bound = DN_max
                        another_pred = fine1[ib, j, i] + coarse2[ib, j, i] - coarse1[ib, j, i]
                        
                        if pred_val < DN_min: pred_val = min(max_bound, max(DN_min, another_pred))
                        if pred_val > DN_max: pred_val = max(min_bound, min(DN_max, another_pred))
                        
                        fine2_sub[ib, j - location[isub, 2], i - location[isub, 0]] = pred_val

        R1, R2 = ind_patch1[isub, 2], ind_patch1[isub, 3]
        C1, C2 = ind_patch1[isub, 0], ind_patch1[isub, 1]
        fine2_full[:, R1:R2 + 1, C1:C2 + 1] = fine2_sub

    valid_mask = (background_whole == 0)
    for ib in range(nb):
        fine2_full[ib, valid_mask] = np.clip(fine2_full[ib, valid_mask], DN_min, DN_max)
        if num_back > 0:
            fine2_full[ib, background_whole == 1] = background

    writeimage(fine2_full, out_path, out_dtype=orig_dtype)
    
    endtime = datetime.datetime.now()
    print(f'   - Done! Time elapsed: {(endtime - starttime).seconds} seconds.')

def main():
    data_dir = r"/home/zhaojiazhen/workspace/STF/STF_matlab/data/spatio_temporal_fusion/ML/private_data/syy_setting-9/test/full"
    out_dir = r"/home/zhaojiazhen/workspace/STF/STF_matlab/results/FSDAF/ML/full"
    value_range = [0, 1]
    
    params = {
        'w': 20, 
        'num_similar_pixel': 20,
        'min_class': 4.0, 
        'max_class': 6.0,
        'num_pure': 100, 
        'scale_factor': 16,    
        'block_size': 30,      
        'background': -9999,
        'background_band': 3,
        'I': 20, 
        'maxStdv': 500, 
        'minDis': 500, 
        'minS': 200, 
        'M': 0.05
    }

    os.makedirs(out_dir, exist_ok=True)
    
    l1_dir = os.path.join(data_dir, "Landsat_01")
    l1_files = sorted(glob.glob(os.path.join(l1_dir, "*.tif")))
    
    if not l1_files:
        print(f"No Landsat_01 tif files found in {l1_dir}")
        return

    for f1_path in l1_files:
        basename = os.path.basename(f1_path)
        prefix = basename.split('_L_')[0] if '_L_' in basename else basename.split('.')[0]
        
        m1_dir = os.path.join(data_dir, "MODIS_01")
        m2_dir = os.path.join(data_dir, "MODIS_02")
        
        m1_cands = glob.glob(os.path.join(m1_dir, f"{prefix}*.tif"))
        m2_cands = glob.glob(os.path.join(m2_dir, f"{prefix}*.tif"))
        
        if not m1_cands or not m2_cands:
            print(f"Skipping {basename}: Cannot find matching MODIS_01 or MODIS_02 pairs.")
            continue
            
        c1_path = m1_cands[0]
        c2_path = m2_cands[0]
        
        out_name = basename.replace('_L_', '_Predict_') if '_L_' in basename else f"{prefix}_Predict.tif"
        out_path = os.path.join(out_dir, out_name)

        if os.path.exists(out_path):
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {out_name} 已存在，跳过处理...")
            continue
        
        print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Starting FSDAF...")
        print(f"  - T1 Fine: {f1_path}")
        print(f"  - T1 Coarse: {c1_path}")
        print(f"  - T2 Coarse: {c2_path}")
        print(f"  - Output : {out_path}")
        
        try:
            run_fsdaf_single_pair(f1_path, c1_path, c2_path, out_path, value_range, params)
        except Exception as e:
            print(f"Error processing {basename}: {str(e)}")

if __name__ == "__main__":
    main()