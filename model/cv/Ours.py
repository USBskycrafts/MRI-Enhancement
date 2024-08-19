import torch.functional as F
import torch.nn as nn
from model.unet import UNet
from model.cv.loss import Loss


class Ours(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(Ours, self).__init__()
        input_dim: int = config.getint("model", "input_dim")
        mask_dim: int = config.getint("model", "mask_dim")
        output_dim: int = config.getint("model", "output_dim")
        self.unet1 = UNet(input_dim, mask_dim)
        self.unet2 = UNet(mask_dim, output_dim)
        self.loss = Loss()

    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data["origin"]
        gt = data["target"]
        g = self.unet1(x)
        y = self.unet2(g * x)
        loss = self.loss(x, g, y, gt)
        return {
            "loss": loss,
            "acc_result": {
                "output": y,
                "target": gt
            },
            "output": y
        }
