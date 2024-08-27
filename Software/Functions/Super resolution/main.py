from HSI_MAT_TRANSFER import mat_to_hsi,hsi_to_mat
from Run_model import get_fin_result
from Bicubic_Interpolation import get_bicubic_mat

def main():
    ori_hsi_path = 'ori_hsidata/'
    ori_mat_path = 'ori_matdata/'
    mid_mat_path = 'mid_matdata/'
    result_mat_path = 'result_matdata/'
    result_hsi_path = 'result_hsidata/'
    scale_factor = 2

    hsi_to_mat(ori_hsi_path,ori_mat_path)
    get_bicubic_mat(ori_mat_path+'now_data.mat',scale_factor)
    get_fin_result(mid_mat_path,result_mat_path)
    mat_to_hsi(ori_hsi_path,result_mat_path,result_hsi_path,scale_factor)

if __name__ == "__main__":
    main()