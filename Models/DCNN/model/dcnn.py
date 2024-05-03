# -*- coding: utf-8 -*- #
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common


class ECN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ECN, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.conv1d = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=1, padding=(0, 0))
        self.Relu = nn.LeakyReLU(0.05)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.Relu(x)
        x = self.conv1d(x)
        x = self.Relu(x)
        return x


class ECNB(nn.Module):
    def __init__(self, in_channels):
        super(ECNB, self).__init__()
        self.ECN = ECN(in_channels, in_channels)
        self.ReLU = nn.LeakyReLU(0.05)

    def forward(self, x):
        e = self.ECN(x)
        x = e + x
        x = self.ReLU(x)
        return x


class RFEB(nn.Module):
    def __init__(self, in_channels):
        super(RFEB, self).__init__()
        self.ECNB_1 = ECNB(in_channels)
        self.ECNB_2 = ECNB(in_channels)
        self.ECNB_3 = ECNB(in_channels)
        self.ECN = ECN(in_channels, in_channels // 2)
        self.ReLU = nn.LeakyReLU(0.05)
        self.conv1d_0 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=(1, 1), stride=1, padding=(0, 0))
        self.conv1d_1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=(1, 1), stride=1, padding=(0, 0))
        self.conv1d_2 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=(1, 1), stride=1, padding=(0, 0))
        self.conv1d_3 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=(1, 1), stride=1, padding=(0, 0))

    def forward(self, x):
        rfeb_0 = self.ReLU(self.conv1d_0(x))
        ecnb_0 = self.ECNB_1(x)
        rfeb_1 = self.ReLU(self.conv1d_1(ecnb_0))
        ecnb_1 = self.ECNB_2(ecnb_0)
        rfeb_2 = self.ReLU(self.conv1d_2(ecnb_1))
        ecnb_2 = self.ReLU(self.ECN(self.ECNB_3(ecnb_1)))
        out = torch.cat([rfeb_0, rfeb_1, rfeb_2, ecnb_2], dim=1)
        out = self.conv1d_3(out)
        return out


class feature_fusion_module(nn.Module):
    def __init__(self, n_colors, n_feats):
        super(feature_fusion_module, self).__init__()
        self.merge = nn.Sequential(nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=(3, 3), stride=1, padding=(1, 1)),
                                   nn.LeakyReLU(0.05),
                                   nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=(3, 3), stride=1, padding=(1, 1)),
                                   nn.LeakyReLU(0.05),
                                   nn.Conv2d(n_feats * 2, n_feats, kernel_size=(1, 1), stride=1, padding=(0, 0)),
                                   nn.LeakyReLU(0.005))
        self.conv1d = nn.Sequential(nn.Conv2d(n_feats * 2, n_feats, kernel_size=(1, 1), stride=1, padding=(0, 0)),
                                    nn.LeakyReLU(0.005))
        self.skip_conv = nn.Conv2d(n_colors, n_feats, kernel_size=(3, 3), stride=1, padding=(1, 1))

    def forward(self, x1, x2, x3, xms):
        y_ex = torch.cat([x2, x3], dim=1)
        y_ex = self.merge(y_ex)
        y_ex = torch.cat([x1, y_ex], dim=1)
        y_ex = self.conv1d(y_ex)
        y = y_ex + self.skip_conv(xms)
        return y


class WeightNet(nn.Module):
    def __init__(self, n_channels, out_channels, n_feats=64, conv=common.default_conv):
        super().__init__()

        self.n_channels = n_channels
        self.inc = nn.Sequential(
            conv(n_channels, n_feats, 3),
            nn.BatchNorm2d(n_feats),
            nn.ReLU(inplace=False),
            conv(n_feats, n_feats, 3),
            nn.BatchNorm2d(n_feats),
            nn.ReLU(inplace=False)
        )

        self.outc = conv(n_feats, out_channels, 3)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x1 = self.inc(x)
        logits = self.outc(x1)
        logits = self.final_pool(logits)
        logits = self.softmax(logits.view(-1))
        return logits


class diffused_image_enhancement(nn.Module):
    def __init__(self, n_colors, n_feats, n_subs, n_ovls, n_distribution=8):
        super(diffused_image_enhancement, self).__init__()
        self.add_noise = common.BatchNoise(noise_min=0, noise_max=10)
        self.weight = WeightNet(n_colors, 1, n_feats)
        self.n_distribution = n_distribution
        # calculate the group number (the number of branch networks)
        self.G = math.ceil((n_colors - n_ovls) / (n_subs - n_ovls))
        # calculate group indices
        self.start_idx = []
        self.end_idx = []
        self.Conv1 = nn.Conv2d(n_subs, n_subs, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.Conv2 = nn.Conv2d(n_subs, n_subs, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.Conv3 = nn.Conv2d(n_subs, n_subs, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.Conv4 = nn.Conv2d(n_subs, n_subs, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.Conv5 = nn.Conv2d(n_subs, n_subs, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.Relu = nn.ReLU(inplace=False)
        self.Sig = nn.Sigmoid()

        for g in range(self.G):
            sta_ind = (n_subs - n_ovls) * g
            end_ind = sta_ind + n_subs
            if end_ind > n_colors:
                end_ind = n_colors
                sta_ind = n_colors - n_subs
            self.start_idx.append(sta_ind)
            self.end_idx.append(end_ind)

    def forward_single(self, x):
        b, c, h, w = x.shape

        y = torch.zeros(b, c, h, w).to(x.device)
        channel_counter = torch.zeros(c).to(x.device)

        for g in range(self.G):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]

            xi = x[:, sta_ind:end_ind, :, :]
            channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1
            out = self.Conv1(xi)
            out = self.Conv2(self.Relu(out))
            out = self.Conv3(self.Relu(out))
            Z = self.Conv5(self.Relu(out))
            M = self.Sig(self.Conv4(self.Relu(out)))
            out = M * out + (1 - M) * Z
            y[:, sta_ind:end_ind, :, :] = y[:, sta_ind:end_ind, :, :] + out

        y = y / channel_counter.unsqueeze(1).unsqueeze(2)
        return y + x

    def forward(self, x):
        b = x.shape[0]
        outs = []
        for i in range(b):
            xi = x[i].unsqueeze(0)

            latents = []
            for ni in range(self.n_distribution):
                single_x = xi
                if i > 0:
                    single_x = self.add_noise(single_x)
                latents.append(single_x)

            latents = torch.cat(latents, dim=0)
            x1 = self.forward_single(latents)
            weight = self.weight(latents).view(-1, 1, 1, 1)
            out = torch.sum(x1 * weight, dim=0, keepdim=True)
            outs.append(out)

        return torch.cat(outs, dim=0)


class reconstruction_module(nn.Module):
    def __init__(self, n_subs, n_ovls, n_colors, n_feats, scale, conv=common.default_conv):
        super(reconstruction_module, self).__init__()

        self.n_feats = n_feats
        self.scale = scale
        self.n_colors = n_colors

        # define head module
        m_head = [nn.Conv2d(n_subs, n_feats, kernel_size=(3, 3), stride=1, padding=(1, 1))]
        self.head = nn.Sequential(*m_head)

        self.B1 = RFEB(in_channels=n_feats)
        self.B2 = RFEB(in_channels=n_feats)
        self.B3 = RFEB(in_channels=n_feats)
        self.B4 = RFEB(in_channels=n_feats)
        self.c = nn.Sequential(nn.Conv2d(n_feats * 4, n_feats, kernel_size=(1, 1), stride=1, padding=(0, 0)),
                               nn.LeakyReLU(0.05))

        # calculate the group number (the number of branch networks)
        self.G = math.ceil((n_colors - n_ovls) / (n_subs - n_ovls))
        # calculate group indices
        self.start_idx = []
        self.end_idx = []

        for g in range(self.G):
            sta_ind = (n_subs - n_ovls) * g
            end_ind = sta_ind + n_subs
            if end_ind > n_colors:
                end_ind = n_colors
                sta_ind = n_colors - n_subs
            self.start_idx.append(sta_ind)
            self.end_idx.append(end_ind)

        self.upsampler = nn.Sequential(*[common.Upsampler(conv, self.scale, self.n_feats, act=False),
                                         nn.Conv2d(self.n_feats, n_feats, kernel_size=(3, 3), stride=1,
                                                   padding=(1, 1))])

        self.fusion = feature_fusion_module(n_colors, n_feats)
        self.final = nn.Sequential(nn.Conv2d(n_feats, n_colors, kernel_size=(3, 3), stride=1, padding=(1, 1)),
                                   nn.LeakyReLU(0.05))
        self.lfid = diffused_image_enhancement(n_colors, n_feats, n_subs, n_ovls)

    def forward(self, x, x_lbp, x_hog):
        b, c, h, w = x.shape
        xms = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        y = torch.zeros(b * 3, self.n_feats, h, w).to(x.device)
        x = torch.cat([x, x_lbp, x_hog], 0)
        for g in range(self.G):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]
            xi = x[:, sta_ind:end_ind, :, :].clone()
            xi = self.head(xi)
            out_B1 = self.B1(xi)
            out_B2 = self.B2(out_B1)
            out_B3 = self.B3(out_B2)
            out_B4 = self.B4(out_B3)
            xi = xi + self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
            y = y + xi
        y = y / self.G
        y = self.upsampler(y)
        y1 = y[0:b, :, :, :]
        y2 = y[b:b * 2, :, :, :]
        y3 = y[b * 2:b * 3, :, :, :]
        y = self.fusion(y1, y2, y3, xms)
        y = self.final(y)
        y = self.lfid(y)
        return y


class DCNN(nn.Module):
    def __init__(self, args):
        super(DCNN, self).__init__()
        self.reconstruct = reconstruction_module(args.n_subs, args.n_ovls, args.n_colors, args.n_feats, args.scale)

    def forward(self, x, x_lbp, x_hog):
        r = self.reconstruct(x, x_lbp, x_hog)
        return r
