import torch
import torch.nn as nn
import torch.nn.parameter as p
import torch.nn.functional as F
from model.unet.unet_parts import *


class Encoder(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(Encoder, self).__init__()
        self.input_dim = config.getint("model", "input_dim")

        self.inc = DoubleConv(self.input_dim, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.kernel = None
        self.token_channel = 786

    def forward(self, data, config, gpu_list, acc_result, mode):
        # data must be a clip of "origin" field
        x1 = self.inc(data)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # shape of x5 is [batch_size, 1024, *, *]

        # make use of dynamic computation graph
        if self.kernel is None:
            self.kernel = p.Parameter(torch.randn(
                self.token_channel, x5.shape[1], x5.shape[2], x5.shape[3]).cuda())
        token = F.conv2d(x5, self.kernel, stride=1, padding=0).squeeze()
        return {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "x5": x5,
            "token": token,
        }


class Decoder(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(Decoder, self).__init__()
        self.output_dim = config.getint("model", "output_dim")
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, self.output_dim)

    def forward(self, data: dict, config, gpu_list, acc_result, mode):
        # data must be the output of encoder
        x1, x2, x3, x4, x5, _ = data.values()
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return {
            "logits": logits
        }
