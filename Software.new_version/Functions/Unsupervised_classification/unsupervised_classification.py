# unsupervised_classification.py

import numpy as np
from spectral import kmeans

def calculate_ndvi(nir_band, red_band):
    nir = nir_band.astype(float)
    red = red_band.astype(float)
    denominator = nir + red
    denominator[denominator == 0] = np.nan  # 避免除以零
    ndvi = (nir - red) / denominator
    return np.nan_to_num(ndvi, nan=0.0)  # 将 NaN 替换为 0.0

def find_Red_NIR_bands(listWavelength):
    R_wavelength = 682.5  # Red (625+740)/2
    NIR_wavelength = 850  # NIR

    rFound = nirFound = False
    rPreDifference = nirPreDifference = float('inf')
    rIndex = nirIndex = 0

    for i, value in enumerate(listWavelength):
        if not rFound:
            difference = abs(value - R_wavelength)
            if difference < rPreDifference:
                rPreDifference = difference
            else:
                rIndex = i - 1
                rFound = True

        if not nirFound:
            difference = abs(value - NIR_wavelength)
            if difference < nirPreDifference:
                nirPreDifference = difference
            else:
                nirIndex = i - 1
                nirFound = True

    return (rIndex, nirIndex)

def load_and_process_hsi_data(hsi_data, wavelengths, k=5, max_iterations=10):
    # 执行 K-Means 聚类，并传入 logger
    m, c = kmeans(hsi_data, k, max_iterations)
    cluster_map = m.reshape(hsi_data.shape[:-1])  # 重塑为 2D 映射（行，列）

    # 找到红色和近红外波段
    red_band_index, nir_band_index = find_Red_NIR_bands(wavelengths)
    red_band = hsi_data[:, :, red_band_index].squeeze()
    nir_band = hsi_data[:, :, nir_band_index].squeeze()
    # print(f"Red band shape: {red_band.shape}")
    # print(f"NIR band shape: {nir_band.shape}")

    # 计算 NDVI
    ndvi = calculate_ndvi(nir_band, red_band)
    # print(f"NDVI shape after calculation: {ndvi.shape}")

    return cluster_map, ndvi
