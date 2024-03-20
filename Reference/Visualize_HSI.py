import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from spectral import *


def find_RGB_bands(listWavelength):
    R_wavelength = 682.5 #(625+740)/2 
    G_wavelength = 532.5  #(495+570)/2 
    B_wavelength = 472.5  #(450+495)/2 
    listlen = len(listWavelength) 
    if listlen < 3:
        print("Error: not a hyperspectral file")
        return
    if listWavelength[0] > B_wavelength or listWavelength[-1] < R_wavelength: # if not fully include RGB bands
        return (round(5*listlen/6), round(listlen/2), round(listlen/6)) # considering edge effect, not use (len, len/2, 1)
    
    rFound = gFound = bFound = False
    rPreDifference = gPreDifference = bPreDifference = float('inf') # previously calculated difference
    rIndex = gIndex = bIndex = 0

    for i, value in enumerate(listWavelength):
        if not rFound:
            difference = abs(value - R_wavelength)
            if difference < rPreDifference:  
                rPreDifference = difference
            else: # when the distance starts to grow bigger, the index is found, and it should be the previous i
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

    参数:
    - image_path: 高光谱图像的路径，图像格式为.mat。
    - band_number: 要可视化的波段编号。
    """
    image_data = loadmat(image_path)
    image = image_data['HSI']  # 确保这里的键与你的数据匹配
    band_image = image[:, :, band_number]

    plt.imshow(band_image, cmap='gray')
    plt.title(f'Band {band_number}')
    plt.colorbar()
    plt.show()

def visualize_hsi_rgb(image_path):
    """
    使用三个波段合成RGB图像并可视化。

    参数:
    - image_path: 高光谱图像的路径，图像格式为.mat。
    - bands: 一个包含三个整数的列表或元组，分别代表红色、绿色和蓝色波段的编号。
    """
    image_data = loadmat(image_path)
    image = image_data['HSI']  # 确保这里的键与你的数据匹配
    wavelengths = image_data['wavelength']  # 假设波长信息存储在'wavelength'键下，根据实际情况调整

    # 使用find_RGB_bands函数找到最接近RGB的波段
    r_band, g_band, b_band = find_RGB_bands(wavelengths.flatten())  # 假设wavelengths是一个二维数组，需要flatten

    # 分别提取R, G, B波段
    r_image = image[:, :, r_band]
    g_image = image[:, :, g_band]
    b_image = image[:, :, b_band]

    # 分别对R, G, B波段进行归一化
    r_image_normalized = r_image / np.max(r_image)
    g_image_normalized = g_image / np.max(g_image)
    b_image_normalized = b_image / np.max(b_image)

    # 合并归一化后的波段
    rgb_image = np.dstack((r_image_normalized, g_image_normalized, b_image_normalized))

    plt.imshow(rgb_image)
    plt.title('RGB Composite')
    plt.show()

# # 使用示例
mat_image_path = 'test_data.mat'
#mat_image_path = 'test_data_interpolated.mat'
# visualize_hsi_band(mat_image_path, 10)  # 显示第10个波段
visualize_hsi_rgb(mat_image_path)