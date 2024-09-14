from .HSI_MAT_TRANSFER import mat_to_hsi,hsi_to_mat
from .Run_model import get_fin_result
from .Bicubic_Interpolation import get_bicubic_mat


def run_super_resolution(ori_hsi_path, ori_mat_path, mid_mat_path, result_mat_path, result_hsi_path, callback=None):
    scale_factor = 2

    hsi_to_mat(ori_hsi_path, ori_mat_path)
    if callback:
        callback("HSI 转换为 MAT 完成")
    try:
        print(ori_mat_path + 'now_data.mat')
        get_bicubic_mat(ori_mat_path + 'now_data.mat',mid_mat_path, scale_factor)
        if callback:
            callback("双三次插值完成")
    except:
        print('双三次插值报错')
        return


    get_fin_result(mid_mat_path, result_mat_path)
    if callback:
        callback("获取最终结果完成")
    mat_to_hsi(ori_hsi_path, result_mat_path, result_hsi_path, scale_factor)
    if callback:
        callback("完成")



if __name__ == "__main__":
    ori_hsi_path = 'ori_hsidata/'
    ori_mat_path = 'ori_matdata/'
    mid_mat_path = 'mid_matdata/'
    result_mat_path = 'result_matdata/'
    result_hsi_path = 'result_hsidata/'
    run_super_resolution(ori_hsi_path,ori_mat_path,mid_mat_path,result_mat_path,result_hsi_path)