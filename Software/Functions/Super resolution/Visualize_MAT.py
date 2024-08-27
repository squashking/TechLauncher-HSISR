import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from spectral import *
import h5py
import os
from PIL import Image

def load_mat_v73(file_path,dataname):
    with h5py.File(file_path, 'r') as file:
        # 假设您要加载的变量名为 'data'
        print(file)
        data = file[dataname][:]
        data = data.squeeze()
        data = data.transpose((2, 1, 0))
    return data


def visualize_hsi_rgb(image_path,tag_name,save_path,save_name):
    try:
        image_data = loadmat(image_path)
        image = image_data[tag_name]
        image = image.squeeze()
    except:
        image = load_mat_v73(image_path,tag_name)
    r_band, g_band, b_band = 289,157,105
    # 分别提取R, G, B波段
    r_image = image[:, :, r_band]
    g_image = image[:, :, g_band]
    b_image = image[:, :, b_band]
    r_image[r_image < 0] = 0
    g_image[g_image < 0] = 0
    b_image[b_image < 0] = 0
    # 分别对R, G, B波段进行归一化
    r_image_normalized = r_image / np.max(r_image)
    g_image_normalized = g_image / np.max(g_image)
    b_image_normalized = b_image / np.max(b_image)

    rgb_image = np.dstack((r_image_normalized, g_image_normalized, b_image_normalized))
    rgb_hsi = (rgb_image * 255).astype(np.uint8)
    im_hsi = Image.fromarray(rgb_hsi)

    output_path = os.path.join(save_path, save_name)
    plt.imsave(output_path, rgb_image)
    plt.imshow(rgb_image)
    plt.title('RGB Composite')
    plt.show()
    im_hsi.save(output_path)

# # 使用示例
#image_path,tag_name,save_path,save_name = 'data/test_path_new/result/result_00013.mat','result','view',"result.png"
#image_path,tag_name,save_path,save_name = 'get_result_data/result.mat','result','Result/SR result',"result.png"

image_path,tag_name,save_path,save_name = 'mid_matdata/mid_data.mat','ms','view',"result_msd2.png"
visualize_hsi_rgb(image_path,tag_name,save_path,save_name)
