import numpy as np
import torch.utils.data as data
import torch
import h5py
from scipy.io import loadmat

class HSResultData(data.Dataset):
    def __init__(self, image_dir, use_3D=False):
        self.use_3Dconv = use_3D
        try:
            with h5py.File(image_dir,'r') as test_data:
                self.ms = np.transpose(np.array(test_data['ms'][...], dtype=np.float32),(3,2,1,0))
                self.lms = np.transpose(np.array(test_data['ms_bicubic'][...], dtype=np.float32),(3,2,1,0))
        except:
            image_data = loadmat(image_dir)
            image_ms = image_data['ms']
            image_ms_bicubic = image_data['ms_bicubic']
            self.ms = image_ms[np.newaxis,:,:,:]
            self.lms = image_ms_bicubic[np.newaxis, ...]



    def __getitem__(self, index):
        ms = self.ms[index, :, :, :]
        lms = self.lms[index, :, :, :]
        if self.use_3Dconv:
            ms, lms = ms[np.newaxis, :, :, :], lms[np.newaxis, :, :, :]
            ms = torch.from_numpy(ms.copy()).permute(0, 3, 1, 2)
            lms = torch.from_numpy(lms.copy()).permute(0, 3, 1, 2)
        else:
            ms = torch.from_numpy(ms.copy()).permute(2, 0, 1)
            lms = torch.from_numpy(lms.copy()).permute(2, 0, 1)
        return ms, lms

    def __len__(self):
        return self.lms.shape[0]

    def get_shape(self):
        return self.ms.shape
