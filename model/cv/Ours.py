from sympy import N
import torch
import torch.functional as F
import torch.nn as nn
from model.unet import UNet
from model.cv.loss import Loss
from tools.accuracy_init import init_accuracy_functions


class Ours(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(Ours, self).__init__()
        input_dim: int = config.getint("model", "input_dim")
        mask_dim: int = config.getint("model", "mask_dim")
        output_dim: int = config.getint("model", "output_dim")
        self.unet1 = UNet(input_dim, mask_dim)
        self.unet2 = UNet(mask_dim + input_dim, output_dim)
        self.loss = Loss()
        self.accuracy_functions = init_accuracy_functions(
            config, *args, **params)

    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data["origin"]
        gt = data["target"]
        g = self.unet1(x)
        y = self.unet2(torch.cat([x, g], dim=1))
        loss = self.loss(x, g, y, gt)
        for name, method in self.accuracy_functions.items():
            if acc_result is None:
                acc_result = {}
            if acc_result.get(name, None) is None:
                acc_result[name] = [method(y, gt, config)]
            else:
                acc_result[name].append(method(y, gt, config))
        return {
            "output": y,
            "target": gt,
            "loss": loss,
            "acc_result": acc_result,
        }
