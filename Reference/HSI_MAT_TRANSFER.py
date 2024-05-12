import spectral.io.envi as envi
import numpy as np
from scipy.io import savemat
from scipy.io import loadmat

def hsi_to_mat(bil_path,header_path, filename):
    hsi = envi.open(header_path, bil_path)
    bands = range(hsi.nbands)
    data = hsi.read_bands(bands)
    data = data.astype(np.float32)
    wavelengths = hsi.metadata.get('wavelength', [])
    mat_dict = {'HSI': data, 'wavelength': wavelengths}
    savemat(filename, mat_dict)

def mat_to_hsi(mat_path,save_path_hsi):
    image_data = loadmat(mat_path)
    image = image_data['HSI']  
    wavelengths = image_data.get('wavelength', None)
    metadata = {'wavelength': wavelengths}
    envi.save_image(save_path_hsi, image, dtype=np.float32, metadata=metadata, interleave='bil', ext='bil',force=True)


bil_path ="data/test_path/0ori_data/2021-03-31--12-56-31_round-0_cam-1_tray-Tray_1.bil"
header_path = "data/test_path/0ori_data/2021-03-31--12-56-31_round-0_cam-1_tray-Tray_1.hdr"
save_path_mat = 'data/test_path/1hst_to_mat/test1.mat'
hsi_to_mat(bil_path,header_path,save_path_mat)

mat_path = 'data/test_path/1hst_to_mat/test1.mat'
save_path_hsi = 'data/test_path/2mat_to_hsi/test1.hdr'
mat_to_hsi(mat_path,save_path_hsi)

