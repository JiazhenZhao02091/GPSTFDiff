import numpy as np
import tifffile
import os


def read_raster(infile):
    data = tifffile.imread(infile)
    if data.ndim == 2:
        rows, cols = data.shape
        data = data.reshape(1, rows, cols)
    elif data.ndim == 3:
        # Assuming the first dimension is the band if it's smaller, 
        # but let's assume it could be (C, H, W) or (H, W, C).
        # We will return it as is, typically (band, row, col)
        if data.shape[0] < data.shape[-1]:
            # It's (C, H, W)
            nb, rows, cols = data.shape
        else:
            # It's (H, W, C) -> transpose to (C, H, W)
            data = np.transpose(data, (2, 0, 1))
            nb, rows, cols = data.shape
    else:
        raise Exception("Unsupported image dimensions")
    return rows, cols, data


def read_raster_new(infile):
    data = tifffile.imread(infile)
    if data.ndim == 2:
        rows, cols = data.shape
        data = data.reshape(rows, cols, 1)
    elif data.ndim == 3:
        # Expected return is (rows, cols, band)
        if data.shape[-1] < data.shape[0]:
            # already (H, W, C)
            rows, cols, nb = data.shape
        else:
            # (C, H, W) -> transpose to (H, W, C)
            data = np.transpose(data, (1, 2, 0))
            rows, cols, nb = data.shape
    else:
        raise Exception("Unsupported image dimensions")
    return rows, cols, data


def writeimage(bands, path, in_ds):
    # 'in_ds' is kept for signature compatibility, but unused since we rely purely on tifffile without GDAL.
    # bands is currently passed as a list of 2D arrays or a 3D numpy array
    if bands is None or len(bands) == 0:
        return
        
    if isinstance(bands, list):
        # Convert list of 2D bands to a 3D array (C, H, W)
        bands_array = np.stack(bands)
    else:
        bands_array = np.array(bands)
        if bands_array.ndim == 2:
            bands_array = bands_array[np.newaxis, :, :]

    # We discard exact GDT_Byte/GDT_Float32 GDAL mappings and rely on numpy/tifffile
    tifffile.imwrite(path, bands_array)

