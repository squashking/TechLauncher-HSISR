import torch
import numpy as np
import torch.utils.data as data
from os import listdir
from os.path import join
import scipy.io as scio
from model.utils_dcnn import degradation_bsrgan
import torch.nn.functional as F
import os
import skimage.feature
from skimage import feature as ft
import random

from multiprocessing import Pool
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
 

def image_feature_extraction_subset(img_list):
    img_lbp_list = []
    img_hog_list = []
    # 使用tqdm在每个子任务中显示进度
    for img in tqdm(img_list, desc="Processing subset"):
        c, H, W = img.shape
        img_lbp = np.zeros([c, H, W])
        img_hog = np.zeros([c, H, W])
        for channel in range(c):
            feature_lbp = skimage.feature.local_binary_pattern(img[channel, :, :], 8, 1.0, method='default')
            feature_hog = ft.hog(img[channel, :, :], orientations=12, pixels_per_cell=[8, 8], cells_per_block=[4, 4],
                                 block_norm='L1', transform_sqrt=True, feature_vector=False, visualize=True)[1]
            feature_hog = np.nan_to_num(feature_hog)
            feature_hog = (feature_hog - np.min(feature_hog)) / (np.max(feature_hog) - np.min(feature_hog) + 1e-6)
            img_lbp[channel, :, :] = feature_lbp
            img_hog[channel, :, :] = feature_hog
        img_lbp_list.append(img_lbp.astype(np.float16))
        img_hog_list.append(img_hog.astype(np.float16))
    return img_lbp_list, img_hog_list

def image_feature_extraction(img_list, num_processes=None):
    # 将 img_list 分割成 num_processes 个子集
    img_list_subsets = np.array_split(img_list, num_processes)
    
    # 使用ProcessPoolExecutor创建进程池
    results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # 移除了使用tqdm创建进度条的代码
        futures = [executor.submit(image_feature_extraction_subset, subset) for subset in img_list_subsets]
        for future in as_completed(futures):
            results.append(future.result())
    
    # 合并结果
    img_lbp_list = [item for sublist in results for item in sublist[0]]
    img_hog_list = [item for sublist in results for item in sublist[1]]
    return img_lbp_list, img_hog_list
    


def image_feature_extraction_test(img_list):
    img_lbp_list = []
    img_hog_list = []
    for img in img_list:
        c, H, W = img.shape
        img_lbp = np.zeros([c, H, W])
        img_hog = np.zeros([c, H, W])
        for channel in tqdm(range(c),desc = "channel"):
        # for channel in range(c):
            feature_lbp = skimage.feature.local_binary_pattern(img[channel, :, :], 8, 1.0, method='default')
            feature_hog = ft.hog(img[channel, :, :], orientations=12, pixels_per_cell=[8, 8], cells_per_block=[4, 4],
                                 block_norm='L1', transform_sqrt=True, feature_vector=False, visualize=True)[1]
            feature_hog = np.nan_to_num(feature_hog)
            feature_hog = (feature_hog - np.min(feature_hog)) / (np.max(feature_hog) - np.min(feature_hog) + 1e-6)
            img_lbp[channel, :, :] = feature_lbp
            img_hog[channel, :, :] = feature_hog
        img_lbp_list.append(img_lbp.astype(np.float32))
        img_hog_list.append(img_hog.astype(np.float32))
    return img_lbp_list, img_hog_list


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.mat'])


def data_augmentation(image_filenames, n_latent, scale):
    image_filenames_imp = []
    label_imp = []
    for i in range(len(image_filenames)):
        mat = scio.loadmat(image_filenames[i], verify_compressed_data_integrity=False)
        label = mat['hr'].astype(np.float32)
        input = F.interpolate(torch.from_numpy(label).unsqueeze(0), scale_factor=1 / scale, mode='bicubic',
                              align_corners=False).squeeze()
        input = np.array(input)
        for i in range(n_latent + 1):
            if i > 0:
                input = degradation_bsrgan(input)
            image_filenames_imp.append(input)
            label_imp.append(label)
    return image_filenames_imp, label_imp


class TrainsetFromFolder(data.Dataset):
    def __init__(self, dataset_dir, n_latent, scale, patch_size):
        super(TrainsetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.image_filenames_imp, self.label = data_augmentation(self.image_filenames, n_latent, scale)
        num_processes = 40
        self.input_lbp, self.input_hog  = image_feature_extraction(self.image_filenames_imp, num_processes)
        # self.input_lbp, self.input_hog = image_feature_extraction(self.image_filenames_imp)
        self.patch_size = patch_size
        self.scale = scale

    def __getitem__(self, index):
        index = index % len(self.input_hog)
        b, h, w = self.input_hog[index].shape
        # ----------------------------
        # randomly crop the patch
        # ----------------------------
        rnd_h = random.randint(0, max(0, h - self.patch_size))
        rnd_w = random.randint(0, max(0, w - self.patch_size))
        hr_hsi_x = int(rnd_h * self.scale)
        hr_hsi_y = int(rnd_w * self.scale)
        input = torch.tensor(self.image_filenames_imp[index].astype(np.float32))[:, rnd_h:rnd_h + self.patch_size,
                rnd_w:rnd_w + self.patch_size]
        input_lbp = torch.tensor(self.input_lbp[index])[:, rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
        input_hog = torch.tensor(self.input_hog[index])[:, rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
        label = torch.tensor(self.label[index])[:, hr_hsi_x:hr_hsi_x + self.patch_size * self.scale,
                hr_hsi_y:hr_hsi_y + self.patch_size * self.scale]

        return input, input_lbp, input_hog, label

    def __len__(self):
        return 64
