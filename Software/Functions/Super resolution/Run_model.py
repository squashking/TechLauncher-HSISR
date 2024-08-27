import os
import time
from scipy.io import savemat
from Get_data import HSResultData
from torch.utils.data import DataLoader
from torchnet import meter
from MSDformer import MSDformer
from common import *

fin_data_dir = './ori_matdata'
result_path = './result_matdata'


def get_fin_result(fin_data_dir,result_path):
    model_name = './weights/fin_msdformer.pth'
    device = torch.device("cpu")
    start_time = time.time()

    for filename in os.listdir(fin_data_dir):
        fin_data = os.path.join(fin_data_dir, filename)
        result_set = HSResultData(fin_data)
        result_loader = DataLoader(result_set, batch_size=1, shuffle=False)
        with torch.no_grad():
            epoch_meter = meter.AverageValueMeter()
            epoch_meter.reset()
            net = MSDformer(n_subs=8, n_ovls=2, n_colors=480, scale=2,
                            n_feats=240, n_DCTM=4, conv=default_conv)
            net.to(device).eval()
            state_dict = torch.load(model_name, map_location=device, weights_only=True)  # 添加 weights_only=True
            net.load_state_dict(state_dict["model"])
            output = []
            for i, (ms, lms) in enumerate(result_loader):
                ms, lms = ms.to(device), lms.to(device)
                y = net(ms, lms, device)  # 传递 device 参数
                y = y.squeeze().cpu().numpy().transpose(1, 2, 0)
                output.append(y)
        save_dir = result_path + '/result_' + os.path.splitext(filename)[0] + '.mat'
        data_dict = {'result': output}
        savemat(save_dir, data_dict)

    end_time = time.time()  # 记录结束时间
    print(f"运行时间: {end_time - start_time} 秒")  # 打印运行时间

if __name__ == "__main__":
    fin_data_dir = './ori_matdata'
    result_path = './result_matdata'
    get_fin_result(fin_data_dir,result_path)