import numpy as np
import torch.utils.data as data
import scipy.io as sio
import torch
import h5py

class HSTestData(data.Dataset):
    def __init__(self, image_dir, use_3D=False):
        with h5py.File(image_dir,'r') as test_data:
        #test_data = sio.loadmat(image_dir)
            self.use_3Dconv = use_3D
            self.ms = np.transpose(np.array(test_data['ms'][...], dtype=np.float32),(3,2,1,0))
            self.lms = np.transpose(np.array(test_data['ms_bicubic'][...], dtype=np.float32),(3,2,1,0))
            self.gt = np.transpose(np.array(test_data['gt'][...], dtype=np.float32),(3,2,1,0))

    def __getitem__(self, index):
        gt = self.gt[index, :, :, :]
        ms = self.ms[index, :, :, :]
        lms = self.lms[index, :, :, :]
        if self.use_3Dconv:
            ms, lms, gt = ms[np.newaxis, :, :, :], lms[np.newaxis, :, :, :], gt[np.newaxis, :, :, :]
            ms = torch.from_numpy(ms.copy()).permute(0, 3, 1, 2)
            lms = torch.from_numpy(lms.copy()).permute(0, 3, 1, 2)
            gt = torch.from_numpy(gt.copy()).permute(0, 3, 1, 2)
        else:
            ms = torch.from_numpy(ms.copy()).permute(2, 0, 1)
            lms = torch.from_numpy(lms.copy()).permute(2, 0, 1)
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        #ms = torch.from_numpy(ms.transpose((2, 0, 1)))
        #lms = torch.from_numpy(lms.transpose((2, 0, 1)))
        #gt = torch.from_numpy(gt.transpose((2, 0, 1)))
        return ms, lms, gt

    def __len__(self):
        return self.gt.shape[0]

    def get_shape(self):
        return self.ms.shape