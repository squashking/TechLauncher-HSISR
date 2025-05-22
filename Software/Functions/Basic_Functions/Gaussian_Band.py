import os
import numpy as np
from scipy.stats import norm
from typing import Tuple, List


def read_wavelengths_from_hdr(hdr_path: str) -> np.ndarray:
    """
    从 .hdr 文件中读取波长信息，返回 shape=(bands,) 的 ndarray。
    """
    if not os.path.exists(hdr_path):
        raise FileNotFoundError(f"HDR file not found: {hdr_path}")

    wavelengths: List[float] = []
    try:
        with open(hdr_path, 'r') as file:
            in_block = False
            for line in file:
                line = line.strip()
                if 'WAVELENGTHS' in line:
                    in_block = True
                    continue
                if 'WAVELENGTHS_END' in line:
                    break
                if in_block:
                    try:
                        wavelengths.append(float(line))
                    except ValueError:
                        continue
    except Exception as e:
        raise RuntimeError(f"Error reading HDR file: {e}")

    if not wavelengths:
        raise ValueError("No wavelengths found in HDR file.")

    return np.array(wavelengths)


def read_bil_data(
        bil_path: str,
        rows: int,
        cols: int,
        bands: int,
        dtype=np.uint16
) -> np.ndarray:
    """
    读取 .bil 文件，返回形状为 (rows, cols, bands) 的高光谱数据。
    """
    if not os.path.exists(bil_path):
        raise FileNotFoundError(f"BIL file not found: {bil_path}")

    try:
        raw = np.fromfile(bil_path, dtype=dtype)
        expected_size = rows * cols * bands
        if raw.size != expected_size:
            raise ValueError(f"File size mismatch: expected {expected_size}, got {raw.size}")
        cube = raw.reshape((rows, bands, cols))
        return np.transpose(cube, (0, 2, 1))  # 转换为 (rows, cols, bands)
    except Exception as e:
        raise RuntimeError(f"Error reading BIL data: {e}")


def gaussian_weighted_rgb(
        hsi_cube: np.ndarray,
        wavelengths: np.ndarray,
        sigma: float = 20,
        r_center: float = 650,
        g_center: float = 550,
        b_center: float = 450
) -> np.ndarray:
    """
    使用高斯加权法将高光谱图像转换为 RGB 图像。
    """

    def weights(center: float) -> np.ndarray:
        w = norm.pdf(wavelengths, center, sigma)
        return w / np.sum(w)

    try:
        R = np.tensordot(hsi_cube, weights(r_center), axes=([2], [0]))
        G = np.tensordot(hsi_cube, weights(g_center), axes=([2], [0]))
        B = np.tensordot(hsi_cube, weights(b_center), axes=([2], [0]))
        rgb = np.stack([R, G, B], axis=-1)
        rgb_norm = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        return rgb_norm
    except Exception as e:
        raise RuntimeError(f"Error during Gaussian RGB conversion: {e}")


def mean_rgb_from_ranges(
        hsi_cube: np.ndarray,
        wavelengths: np.ndarray,
        r_range: Tuple[int, int] = (640, 700),
        g_range: Tuple[int, int] = (520, 580),
        b_range: Tuple[int, int] = (450, 500)
) -> np.ndarray:
    """
    使用波段区间平均法将高光谱图像转换为 RGB 图像。
    """

    def band_range(wrange: Tuple[int, int]) -> np.ndarray:
        indices = np.where((wavelengths >= wrange[0]) & (wavelengths <= wrange[1]))[0]
        if len(indices) == 0:
            raise ValueError(f"No bands found in range {wrange}")
        return indices

    try:
        r_idx = band_range(r_range)
        g_idx = band_range(g_range)
        b_idx = band_range(b_range)

        R = np.mean(hsi_cube[:, :, r_idx], axis=2)
        G = np.mean(hsi_cube[:, :, g_idx], axis=2)
        B = np.mean(hsi_cube[:, :, b_idx], axis=2)

        rgb = np.stack([R, G, B], axis=-1)
        rgb_norm = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        return rgb_norm
    except Exception as e:
        raise RuntimeError(f"Error during Mean RGB conversion: {e}") 