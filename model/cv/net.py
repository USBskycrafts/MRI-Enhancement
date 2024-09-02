from re import I
import torch
import torch.nn as nn
import itertools
from tools.accumulate_tool import accumulate_cv_data, accumulate_loss
from .parts import *


class ProposedModel(nn.Module):
    def __init__(self, config, gpu_list, *args, **params) -> None:
        super().__init__()
        # the dimension after data augmentation
        self.input_dim = config.getint("model", "input_dim")
        # output should be the dimension of the original data
        self.output_dim = config.getint("model", "output_dim")
        # writer = params["writer"]

        self.discriminator = Discriminator(self.output_dim)
        self.generator = Generator(
            self.input_dim, self.output_dim, self.discriminator)

    def forward(self, data, config, gpu_list, acc_result, mode):
        t1, t2, t1ce = data["t1"], data["t2"], data["t1ce"]
        loss = 0
        _loss, fake = self.generator({
            "T1": t1,
            "T2": t2,
            "T1CE": t1ce
        }).values()
        loss += _loss

        _loss, fake_label, real_label = self.discriminator({
            "fake": fake,
            "real": t1ce
        }).values()
        loss += _loss
        acc_result = accumulate_cv_data({
            "output": fake.detach(),
            "gt": t1ce.detach(),
        }, acc_result, config, mode)

        return {
            "loss": loss,
            "acc_result": acc_result,
            "output": list(torch.split(fake.squeeze().cpu().detach(), 1, dim=0))
        }
