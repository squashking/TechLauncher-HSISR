import spectral.io.envi as envi
import numpy as np
from scipy.io import savemat
from scipy.io import loadmat
import h5py

def load_mat_v73(file_path,dataname):
    with h5py.File(file_path, 'r') as file:
        data = file[dataname][:]
        data = data.squeeze()
        data = data.transpose((2, 1, 0))
    return data

def hsi_to_mat(bil_path,header_path, filename):
    hsi = envi.open(header_path, bil_path)
    bands = range(hsi.nbands)
    data = hsi.read_bands(bands)
    data = data.astype(np.float32)
    wavelengths = hsi.metadata.get('wavelength', [])
    mat_dict = {'HSI': data, 'wavelength': wavelengths}
    savemat(filename, mat_dict)

def mat_to_hsi(mat_path,save_path_hsi,tag):
    try:
        image_data = loadmat(mat_path)
        image = image_data[tag]
        image = image.squeeze()
    except:
        image = load_mat_v73(mat_path,tag)
    image_new = []
    for i in range(480):
        image_bond = image[:, :, i]
        image_bond[image_bond < 0] = 0
        image_bond = image_bond / np.max(image_bond)
        image_bond = np.round(image_bond * 4095)
        image_new.append(image_bond)
    image = np.stack(image_new,axis=2)
    envi.save_image(save_path_hsi, image, dtype=np.float32, interleave='bil', ext='bil',force=True)
    headerPath = 'get_result_data/2024-04-26--17-22-00_round-0_cam-1_tray-1.hdr'
    dictMeta = read_PSI_header(headerPath)
    create_envi_header(save_path_hsi, dictMeta)

def read_PSI_header(filePath):
    data_dict = {}
    with open(filePath, "r") as file:
        lines = file.readlines()
    wavelengths = []
    for line in lines:
        parts = line.strip().split(" ")
        if "WAVELENGTHS" in parts:
            reading_wavelengths = True
            continue
        elif "WAVELENGTHS_END" in parts:
            reading_wavelengths = False
            data_dict["WAVELENGTHS"] = wavelengths
            continue
        if len(parts) == 2:
            key, value = parts
            data_dict[key] = value
        elif reading_wavelengths:
            wavelengths.append(float(parts[0]))
    return data_dict


def create_envi_header(filename, dictMeta):
    with open(filename, 'w') as file:
        file.write("ENVI\n")
        file.write("bands = {}\n".format(dictMeta['NBANDS']))
        file.write("byte order = 0\n")
        #file.write("data type = {}\n".format(dictMeta['NBITS']))
        file.write("data type = 4\n")
        file.write("file type = ENVI Standard\n")
        file.write("header offset = 0\n")
        file.write("interleave = {}\n".format(dictMeta['LAYOUT'].lower()))
        file.write("lines = {}\n".format(str(350)))
        file.write("samples = {}\n".format(str(500)))
        file.write("wavelength units = nm\n")
        file.write("wavelength = {")
        file.write(','.join(map(str, dictMeta['WAVELENGTHS'])))
        file.write("}\n")

mat_path = 'data/server_data/00005.mat'
save_path_hsi = 'Result_flower/Original low resolution HSI file/LR.hdr'
tag = 'ms'
mat_to_hsi(mat_path,save_path_hsi,tag)

