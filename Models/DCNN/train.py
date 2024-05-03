# coding:utf-8
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from option import opt
from data_utils import TrainsetFromFolder
from torch.optim.lr_scheduler import LambdaLR
from data_utils import is_image_file
from os import listdir
from eval import PSNR, SSIM, SAM
import numpy as np
import scipy.io as scio
from model.dcnn import DCNN
from data_utils import image_feature_extraction,image_feature_extraction_test
import math
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def main():
    if opt.cuda:
        print("=> Use GPU ID: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    # Loading datasets
    train_set = TrainsetFromFolder(opt.train_path, opt.n_latent, opt.scale, opt.patch_size)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    # Buliding model
    model = DCNN(opt)
    L1_loss = torch.nn.L1Loss()
    if opt.cuda:
        model = nn.DataParallel(model).cuda()
        L1_loss = L1_loss.cuda()
    else:
        model = model.cpu()
    print('# parameters:', sum(param.numel() for param in model.parameters()))

    # Setting Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08)

    # optionally resuming from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Setting learning rate
    lf = lambda x: ((1 + math.cos(x * math.pi / opt.nEpochs)) / 2) * (1 - opt.lrf) + opt.lrf  # cosine
    scheduler = LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = MultiStepLR(optimizer, milestones=[100, 150, 200], gamma=0.1, last_epoch=-1)

    # Training
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))
        train(train_loader, optimizer, model, L1_loss, epoch)
        if epoch % 10 == 0:  # 每10个epoch进行一次测试和保存检查点
            test(opt.test_path, model)
            save_checkpoint(epoch, model, optimizer)
        scheduler.step()
    # for epoch in range(opt.start_epoch, opt.nEpochs + 1):
    #     print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))
    #     train(train_loader, optimizer, model, L1_loss, epoch)
    #     test(opt.test_path, model)
    #     scheduler.step()
    #     save_checkpoint(epoch, model, optimizer)


def train(train_loader, optimizer, model, L1_loss, epoch):
    model.train()
    for iteration, batch in enumerate(train_loader, 1):
        input, input_lbp, input_hog, label = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(
            batch[3], requires_grad=False)

        if opt.cuda:
            input = input.cuda()
            input_lbp = input_lbp.cuda()
            input_hog = input_hog.cuda()
            label = label.cuda()
        optimizer.zero_grad()
        SR_rough = model(input, input_lbp, input_hog)
        loss = L1_loss(SR_rough, label)
        loss.backward()
        optimizer.step()
        if iteration % int(len(train_loader) // 3) == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(train_loader), loss.item()))
            # break


def test(test_path, model):
    input_path = test_path
    PSNRs = 0
    SSIMs = 0
    SAMs = 0
    model.eval()
    images_name = [x for x in listdir(input_path) if is_image_file(x)]
    for index in range(len(images_name)):
        mat = scio.loadmat(input_path + '/' + images_name[index])
        label = mat['hr'].astype(np.float32)
        input = F.interpolate(torch.from_numpy(label).float().unsqueeze(0), scale_factor=1 / opt.scale, mode='bicubic',
                              align_corners=False).squeeze()

        # hyperLR = mat['lr'].transpose(2, 0, 1).astype(np.float32)
        with torch.no_grad():
            input = Variable(input).contiguous().view(1, -1, input.shape[1], input.shape[2])
            input_lbp, input_hog = image_feature_extraction_test([np.array(input.squeeze())])
            #input_lbp, input_hog = image_feature_extraction([np.array(input.squeeze())])
            input_lbp = torch.from_numpy(input_lbp[0]).unsqueeze(0)
            input_hog = torch.from_numpy(input_hog[0]).unsqueeze(0)
            if opt.cuda:
                input = input.cuda()
                input_lbp = input_lbp.cuda()
                input_hog = input_hog.cuda()
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
            print(
                "===The {}-th picture=====PSNR:{:.3f}=====SSIM:{:.4f}=====SAM:{:.3f}====Name:{}".format(index + 1, psnr,
                                                                                                        ssim, sam,
                                                                                                        images_name[
                                                                                                            index]))
    print("=====averPSNR:{:.3f}=====averSSIM:{:.4f}=====averSAM:{:.3f}".format(PSNRs / len(images_name),
                                                                               SSIMs / len(images_name),
                                                                               SAMs / len(images_name)))


def save_checkpoint(epoch, model, optimizer):
    model_out_path = "./checkpoint/" + opt.save_name + "/{}_reconstruct_model_{}_epoch_{}.pth".format(opt.datasetName,
                                                                                                      opt.scale, epoch)
    state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}
    if not os.path.exists("./checkpoint/" + opt.save_name):
        os.makedirs("./checkpoint/" + opt.save_name)
    torch.save(state, model_out_path)


if __name__ == "__main__":
    main()
