from HSI_MAT_TRANSFER import mat_to_hsi,hsi_to_mat
from Run_model import get_fin_result

def main():
    ori_hsi_path = 'ori_hsidata/'
    ori_mat_path = 'ori_matdata/'
    result_mat_path = 'result_matdata'
    result_hsi_path = 'result_hsidata'

    hsi_to_mat(ori_hsi_path,ori_mat_path)
    get_fin_result(ori_mat_path,result_mat_path)
    #mat_to_hsi(ori_hsi_path,result_mat_path,result_hsi_path)

if __name__ == "__main__":
    main()