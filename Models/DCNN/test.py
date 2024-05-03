import os
import numpy as np
import torch
from os import listdir
import torch.nn as nn
from torch.autograd import Variable
from option import opt
from data_utils import is_image_file
from model.dcnn import DCNN
import scipy.io as scio
from eval import PSNR, SSIM, SAM
from data_utils import image_feature_extraction,image_feature_extraction_test
import torch.nn.functional as F


def main():
    input_path = opt.test_path
    PSNRs = 0
    SSIMs = 0
    SAMs = 0
    if opt.cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
    model = DCNN(opt)
    if opt.cuda:
        model = nn.DataParallel(model).cuda()
    checkpoint = torch.load(opt.model_name)
    model.load_state_dict(checkpoint["model"])
    model = model.module
    model.eval()
    images_name = [x for x in listdir(input_path) if is_image_file(x)]
    for index in range(len(images_name)):
        mat = scio.loadmat(input_path + '/' + images_name[index])
        label = mat['hr'].astype(np.float32)
        input = F.interpolate(torch.from_numpy(label).float().unsqueeze(0), scale_factor=1 / opt.scale, mode='bicubic',
                              align_corners=False).squeeze()
        with torch.no_grad():
            input = Variable(input).contiguous().view(1, -1, input.shape[1], input.shape[2])
            print(input.dtype)
            input_lbp, input_hog = image_feature_extraction_test([np.array(input.squeeze())])
            input_lbp = torch.from_numpy(input_lbp[0]).unsqueeze(0)
            input_hog = torch.from_numpy(input_hog[0]).unsqueeze(0)
            if opt.cuda:
                input = input.cuda()
                input_lbp = input_lbp.cuda()
                input_hog = input_hog.cuda()
            print(input.shape)
            print(input_lbp.shape)
            print(input_hog.shape)
            output = model(input, input_lbp, input_hog)
            HR = mat['hr'].astype(np.float32)
            SR = output.cpu().data[0].numpy().astype(np.float32)
            SR[SR < 0] = 0
            SR[SR > 1.] = 1.
            psnr = PSNR(SR, HR)
            ssim = SSIM(SR, HR)
            sam = SAM(SR, HR)
            PSNRs += psnr
            SSIMs += ssim
            SAMs += sam
            output_path = "./data/result/"
            result_file_path = os.path.join(output_path, images_name[index])
            scio.savemat(result_file_path, {'result': SR})
            print(
                "===The {}-th picture=====PSNR:{:.3f}=====SSIM:{:.4f}=====SAM:{:.3f}====Name:{}".format(index + 1, psnr,
                                                                                                        ssim, sam,
                                                                                                        images_name[
                                                                                                            index]))
    print("=====averPSNR:{:.3f}=====averSSIM:{:.4f}=====averSAM:{:.3f}".format(PSNRs / len(images_name),
                                                                               SSIMs / len(images_name),
                                                                               SAMs / len(images_name)))


if __name__ == "__main__":
    main()
