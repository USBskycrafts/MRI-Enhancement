from collections import OrderedDict
import logging
import torch
import torch.nn as nn
import torch.nn.parameter as p
import torch.nn.functional as F
from typing import OrderedDict
from model.unet.unet_parts import *


class Encoder(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(Encoder, self).__init__()
        self.input_dim = config.getint("model", "input_dim")
        self.gpu_list = gpu_list

        self.inc = DoubleConv(self.input_dim, self.input_dim * 16)
        self.down1 = Down(self.input_dim * 16, self.input_dim * 32)
        self.down2 = Down(self.input_dim * 32, self.input_dim * 64)
        self.down3 = Down(self.input_dim * 64, self.input_dim * 128)
        self.down4 = Down(self.input_dim * 128, self.input_dim * 256)
        self.logger = logging.getLogger("Encoder")

    def forward(self, data, config, gpu_list, acc_result, mode):
        # data must be a clip of "origin" field
        x1 = self.inc(data)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # shape of x5 is [batch_size, 1024, *, *]

        return {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "x5": x5,
        }


class Decoder(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(Decoder, self).__init__()
        self.output_dim = params["output_dim"]
        self.up1 = Up(self.output_dim * 256, self.output_dim * 128)
        self.up2 = Up(self.output_dim * 128, self.output_dim * 64)
        self.up3 = Up(self.output_dim * 64, self.output_dim * 32)
        self.up4 = Up(self.output_dim * 32, self.output_dim * 16)
        self.outc = OutConv(self.output_dim * 16, self.output_dim)

    def forward(self, data: dict, config, gpu_list, acc_result, mode):
        # data must be the output of encoder
        x1, x2, x3, x4, x5 = data.values()
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
