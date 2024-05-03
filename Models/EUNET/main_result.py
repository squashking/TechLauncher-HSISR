import os
import numpy as np
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

from option import args
from data import HSRData_result
from models.network_eunet import EUNet
from utils import utils_model
from utils import utils_image as util
from scipy.io import savemat

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('===> Loading resultset')
    test_path = args.dir_data + 'result'
    test_set = HSRData_result(data_dir=test_path, sigma=args.sigma, augment=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    print('===> Loading model')
    assert os.path.exists(args.model_path), 'Error: model_path is empty.'
    print(f'loading model from {args.model_path}')

    model = define_model(args)
    model = model.to(device)
    model.eval()

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['sam'] = []

    for i, test_data in enumerate(test_loader):
        filename = test_data['name'][0].split("/")[-1]
        
        # print(test_data['L'].shape)
        # print(test_data['S'].shape)
        L = test_data['L'].to(device)
        S = test_data['S'].to(device)

        E = model(L, S)

        E = E.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
        E = np.clip(E, 0., 1.)
        
        save_dir =  './result/result_'+  os.path.splitext(filename)[0] + '.mat'
        data_dict = {'result': E}
        savemat(save_dir, data_dict)
        



def define_model(args):
    model = EUNet(scale=args.scale, n_iter=args.n_iters, n_colors=args.n_colors, n_feats=args.n_feats,
                  n_modules=args.n_modules, block=args.block_type, n_blocks=args.n_blocks, dilations=args.dilations,
                  expand_ratio=args.expand_ratio, is_blur=args.is_blur)

    param_key_g = 'params'

    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
                          strict=True)

    return model


if __name__ == '__main__':
    main()