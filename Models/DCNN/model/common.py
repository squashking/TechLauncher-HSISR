import math
import torch
import torch.nn as nn
import numpy as np

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class BatchNoise(object):
    def __init__(self, noise=0, noise_min=5, noise_max=25):
        super(BatchNoise, self).__init__()
        self.noise = noise
        self.noise_min = noise_min
        self.noise_max = noise_max

    def __call__(self, x, train=True):
        # x -> [0, 1]
        b, c, h, w = x.size()
        device = x.device
        if train:
            noise_level = np.random.randint(self.noise_min, self.noise_max)
        else:
            noise_level = self.noise
        noise_level = torch.rand(b, 1, 1, 1).to(device) * (noise_level / 255.)
        noise = torch.randn_like(x).to(device) * noise_level
        x = x + noise
        return x
