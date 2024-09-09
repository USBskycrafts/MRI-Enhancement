from re import I
import torch
import torch.nn as nn
import itertools
from tools.accumulate_tool import accumulate_cv_data, accumulate_loss
from .parts import *
from tensorboardX import SummaryWriter


class ProposedModel(nn.Module):
    def __init__(self, config, gpu_list, *args, **params) -> None:
        super().__init__()
        # the dimension after data augmentation
        self.input_dim = config.getint("model", "input_dim")
        # output should be the dimension of the original data
        self.output_dim = config.getint("model", "output_dim")
        # writer = params["writer"]

        self.generator = Generator(
            self.input_dim, self.output_dim)

    def forward(self, data, config, gpu_list, acc_result, mode, local=None):
        t1, t2, t1ce = data["t1"], data["t2"], data["t1ce"]
        loss = 0
        g_loss, fake = self.generator({
            "T1": t1,
            "T2": t2,
            "T1CE": t1ce
        }, mode).values()
        loss += g_loss

        acc_result = accumulate_cv_data({
            "output": fake.detach(),
            "gt": t1ce.detach(),
        }, acc_result, config, mode)

        if local != None:
            writer: SummaryWriter = local.writer
            if mode == "train":
                global_step = local.global_step
                writer.add_scalar("loss/generator", float(g_loss),
                                  global_step=global_step)
                writer.add_scalar(f"{mode}/psnr", acc_result["PSNR"][-1],
                                  global_step=global_step)
                writer.add_scalar(f"{mode}/ssim", acc_result["SSIM"][-1],
                                  global_step=global_step)
            elif mode == "valid":
                eval_step = local.eval_step
                writer.add_scalar(f"{mode}/psnr", acc_result["PSNR"][-1],
                                  global_step=eval_step)
                writer.add_scalar(f"{mode}/ssim", acc_result["SSIM"][-1],
                                  global_step=eval_step)

        return {
            "loss": loss,
            "acc_result": acc_result,
            "generated": [fake.detach()],
            "output": [(torch.min(fake).detach().item(),
                        torch.max(fake).detach().item(),
                        torch.mean(fake).detach().item())]
        }
