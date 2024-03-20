import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm
from scipy.io import loadmat, savemat

def bicubic_interpolation_hyperspectral(image_path, scale_factor):
    """
    对高光谱图像进行双三次插值，并显示进度条，并保存wavelength信息。

    参数:
    - image_path: 高光谱图像的路径，图像格式为.mat。
    - scale_factor: 缩放因子，大于1表示放大，小于1表示缩小。

    返回:
    - 缩放后的高光谱图像。
    """
    # 从.mat文件中加载图像
    image_data = loadmat(image_path)
    # 假设图像数据存储在名为'HSI'的键中
    image = image_data['HSI']
    # 假设wavelength信息存储在名为'wavelength'的键中
    wavelengths = image_data.get('wavelength', None)

    interpolated_images = []
    # 使用tqdm添加进度条
    for i in tqdm(range(image.shape[2]), desc="Interpolating"):
        band = image[:, :, i]
        interpolated_band = zoom(band, scale_factor, order=3)  # 双三次插值
        interpolated_images.append(interpolated_band)

    interpolated_image = np.stack(interpolated_images, axis=2)

    # 将插值后的图像和wavelength信息保存为.mat文件
    output_path = image_path.replace('.mat', '_interpolated.mat')
    # 确保在保存时包含wavelength信息
    save_data = {'HSI': interpolated_image}
    if wavelengths is not None:
        save_data['wavelength'] = wavelengths
    savemat(output_path, save_data)

    return interpolated_image

# 使用示例
mat_image_path = 'test_data.mat'
scale_factor = 2
interpolated_image = bicubic_interpolation_hyperspectral(mat_image_path, scale_factor)