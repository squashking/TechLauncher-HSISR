import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import spectral.io.envi as envi

def find_RGB_bands(listWavelength):
    R_wavelength = 682.5  # (625+740)/2
    G_wavelength = 532.5  # (495+570)/2
    B_wavelength = 472.5  # (450+495)/2
    listlen = len(listWavelength)
    if listlen < 3:
        print("Error: not a hyperspectral file")
        return
    if listWavelength[0] > B_wavelength or listWavelength[-1] < R_wavelength:  # if not fully include RGB bands
        return (round(5 * listlen / 6), round(listlen / 2),
                round(listlen / 6))  # considering edge effect, not use (len, len/2, 1)

    rFound = gFound = bFound = False
    rPreDifference = gPreDifference = bPreDifference = float('inf')  # previously calculated difference
    rIndex = gIndex = bIndex = 0

    for i, value in enumerate(listWavelength):
        if not rFound:
            difference = abs(value - R_wavelength)
            if difference < rPreDifference:
                rPreDifference = difference
            else:  # when the distance starts to grow bigger, the index is found, and it should be the previous i
                rIndex = i - 1
                rFound = True

        if not gFound:
            difference = abs(value - G_wavelength)
            if difference < gPreDifference:
                gPreDifference = difference
            else:
                gIndex = i - 1
                gFound = True

        if not bFound:
            difference = abs(value - B_wavelength)
            if difference < bPreDifference:
                bPreDifference = difference
            else:
                bIndex = i - 1
                bFound = True

    return (rIndex, gIndex, bIndex)


def read_PSI_header(filePath):
    data_dict = {}
    # Open the text file
    with open(filePath, "r") as file:
        lines = file.readlines()
    # Parse the lines and store values in the dictionary
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

    # Display the parsed data
    return data_dict


# create an ENVI header file for PSI images
def create_envi_header(filename, dictMeta):
    with open(filename, 'w') as file:
        file.write("ENVI\n")
        file.write("description = {Generated by Python}\n")
        file.write("bands = {}\n".format(dictMeta['NBANDS']))
        file.write("byte order = 0\n")
        file.write("data type = {}\n".format(dictMeta['NBITS']))
        file.write("file type = ENVI Standard\n")
        file.write("header offset = 0\n")
        file.write("interleave = {}\n".format(dictMeta['LAYOUT'].lower()))
        file.write("lines = {}\n".format(dictMeta['NROWS']))
        file.write("samples = {}\n".format(dictMeta['NCOLS']))
        # file.write("sensor type = Unknown\n")

        file.write("wavelength units = nm\n")
        file.write("wavelength = {")
        # for wavelength in dictMeta['WAVELENGTHS']:
        # file.write("{:.2f}, ".format(wavelength))
        file.write(','.join(map(str, dictMeta['WAVELENGTHS'])))
        file.write("}\n")

def load_image(image_path, headerPath):
    # check if it's PSI image format
    with open(headerPath, "r") as file:
        first_line = file.readline().strip()
    if first_line.startswith("BYTEORDER"):  # PSI format
        dictMeta = read_PSI_header(headerPath)
        headerPath = header_path
        create_envi_header(headerPath, dictMeta)

    hsi = envi.open(headerPath, image_path)
    return hsi

def normalize_data(data):
    """将数据归一化到0-1范围"""
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def show_cube(data, save_path):
    # 确保数据是三维的
    if len(data.shape) != 3:
        raise ValueError("数据必须是三维的")
    print("原始数据形状:", data.shape)
    # 提取指定波段的数据
    bands = list(find_RGB_bands([float(i) for i in data.metadata['wavelength']]))
    r_band, g_band, b_band = bands[0],bands[1],bands[2]
    # 分别提取R, G, B波段
    r_image = data[:, :, r_band]
    g_image = data[:, :, g_band]
    b_image = data[:, :, b_band]
    r_image[r_image < 0] = 0
    g_image[g_image < 0] = 0
    b_image[b_image < 0] = 0
    # 分别对R, G, B波段进行归一化
    r_image_normalized = r_image / np.max(r_image)
    g_image_normalized = g_image / np.max(g_image)
    b_image_normalized = b_image / np.max(b_image)

    rgb_data = np.dstack((r_image_normalized, g_image_normalized, b_image_normalized))
    # 创建图形和3D轴
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 获取数据的维度
    x_dim, y_dim, z_dim = data.shape
    # 创建网格
    X, Y = np.meshgrid(np.arange(y_dim), np.arange(x_dim))

    # 绘制顶面
    ax.plot_surface(X, Y, np.full((x_dim, y_dim), z_dim),
                    facecolors=rgb_data, rstride=1, cstride=1, shade=True)

    # 绘制侧面
    side1 = normalize_data(data[:, y_dim-100, :])
    side2 = normalize_data(data[x_dim-100, :, :])

    X_side1, Z_side1 = np.meshgrid(np.arange(x_dim), np.arange(z_dim))
    Y_side2, Z_side2 = np.meshgrid(np.arange(y_dim), np.arange(z_dim))

    # 调整 side1 和 side2 的形状
    side1 = np.squeeze(side1).T
    side2 = np.squeeze(side2).T

    ax.plot_surface(np.full_like(X_side1, y_dim), X_side1, Z_side1,
                    facecolors=plt.cm.viridis(side1), rstride=1, cstride=1, shade=True)
    ax.plot_surface(Y_side2, np.full_like(Y_side2, x_dim), Z_side2,
                    facecolors=plt.cm.viridis(side2), rstride=1, cstride=1, shade=True)

    # 设置轴标签和范围
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Wavelength')
    ax.set_xlim(0, y_dim)
    ax.set_ylim(0, x_dim)
    ax.set_zlim(0, z_dim)

    # 设置视角
    ax.view_init(elev=30, azim=45)
    plt.title('Hypercube Visualization')
    plt.show()
    # if len(save_path) > 0:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # else:
    #     plt.show()
    #
    # plt.close(fig)


# 使用示例
if __name__ == "__main__":
    header_path = "Data/2021-03-31--12-56-31_round-0_cam-1_tray-Tray_1.hdr"
    bil_path = "Data/2021-03-31--12-56-31_round-0_cam-1_tray-Tray_1.bil"
    save_path = "Result.jpg"
    hsi = load_image(bil_path, header_path)
    # show_rgb(hsi, save_path)
    show_cube(hsi,save_path)
