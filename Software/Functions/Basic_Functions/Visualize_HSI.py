import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def find_RGB_bands(listWavelength):
    R_wavelength = 682.5  # (625+740)/2
    G_wavelength = 532.5  # (495+570)/2
    B_wavelength = 472.5  # (450+495)/2
    listlen = len(listWavelength)
    if listlen < 3:
        print("Error: not a hyperspectral file")
        return
    if listWavelength[0] > B_wavelength or listWavelength[-1] < R_wavelength:
        return (round(5*listlen/6), round(listlen/2), round(listlen/6))

    rFound = gFound = bFound = False
    rPreDifference = gPreDifference = bPreDifference = float('inf')
    rIndex = gIndex = bIndex = 0

    for i, value in enumerate(listWavelength):
        if not rFound:
            difference = abs(value - R_wavelength)
            if difference < rPreDifference:
                rPreDifference = difference
            else:
                rIndex = i-1
                rFound = True

        if not gFound:
            difference = abs(value - G_wavelength)
            if difference < gPreDifference:
                gPreDifference = difference
            else:
                gIndex = i-1
                gFound = True

        if not bFound:
            difference = abs(value - B_wavelength)
            if difference < bPreDifference:
                bPreDifference = difference
            else:
                bIndex = i-1
                bFound = True

    return (rIndex, gIndex, bIndex)


def visualize_hsi_band(image_path, band_number):
    """
    可视化高光谱图像的单个波段。
    """
    image_data = loadmat(image_path)
    image = image_data['HSI']
    band_image = image[:, :, band_number]

    plt.imshow(band_image, cmap='gray')
    plt.title(f'Band {band_number}')
    plt.colorbar()
    plt.show()


def visualize_hsi_rgb(image_path):
    """
    使用高斯加权法将高光谱图像转换为 RGB 图像并可视化。
    """
    from Gaussian_Band import gaussian_weighted_rgb, read_wavelengths_from_hdr

    image_data = loadmat(image_path)
    image = image_data['HSI']
    wavelengths = image_data['wavelength'].flatten()

    rgb_image = gaussian_weighted_rgb(image, wavelengths)

    plt.imshow(rgb_image)
    plt.title('Gaussian Weighted RGB')
    plt.axis('off')
    plt.show()


# 示例调用
mat_image_path = 'test_data.mat'
# visualize_hsi_band(mat_image_path, 10)
visualize_hsi_rgb(mat_image_path)