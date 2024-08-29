from re import I
import torch
import torch.nn as nn
import itertools
from tools.accumulate_tool import accumulate_cv_data
from .parts import Encoder, Decoder


class ProposedModel(nn.Module):
    def __init__(self, config, gpu_list, *args, **params) -> None:
        super().__init__()
        # the dimension after data augmentation
        self.input_dim = config.getint("model", "input_dim")
        # the augmentation method number
        self.aug_num = config.getint("model", "aug_num")
        # output should be the dimension of the original data
        self.output_dim = config.getint("model", "output_dim")

        # encoder and decoder are the skeleton of the model
        self.encoder = Encoder(config, gpu_list, *args, **params)
        self.decoder1 = Decoder(config, gpu_list, *args,
                                output_dim=self.input_dim, **params)
        self.decoder2 = Decoder(config, gpu_list, *args,
                                output_dim=self.input_dim, **params)
        self.loss = nn.ModuleList([nn.L1Loss()
                                  for _ in range(4)])

        self.mlp = nn.Sequential(
            nn.Linear(self.output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.cross_entropy = nn.ModuleList(
            [nn.CrossEntropyLoss() for _ in range(3)]
        )

        self.enhancement = nn.ModuleList(
            [Encoder(config, gpu_list, *args, **params),
             Decoder(config, gpu_list, *args, output_dim=self.input_dim, **params)]
        )

    def forward(self, data, config, gpu_list, acc_result, mode):
        t1, t2, t1ce = data["t1"], data["t2"], data["t1ce"]
        loss = 0
        data = self.encoder(t2, config, gpu_list, acc_result, mode)
        NH_T2 = self.decoder1(data, config, gpu_list, acc_result, mode)
        T2 = self.decoder2(data, config, gpu_list, acc_result, mode)
        loss += self.loss[0](NH_T2 * torch.exp(-T2), t2)
        data = self.encoder(t1, config, gpu_list, acc_result, mode)
        NH_T1 = self.decoder1(data, config, gpu_list, acc_result, mode)
        T1 = self.decoder2(data, config, gpu_list, acc_result, mode)
        loss += self.loss[1](NH_T1 * (1 - torch.exp(-T1)), t1)
        loss += self.loss[2](NH_T1, NH_T2)

        data = self.enhancement[0](T1, config, gpu_list, acc_result, mode)
        T1CE = self.enhancement[1](data, config, gpu_list, acc_result, mode)
        fake = NH_T1 * (1 - torch.exp(-T1CE))
        loss += self.loss[3](fake, t1ce) * 0.1

        # discriminator
        d_fake = fake.detach().view(-1, self.output_dim)
        d_real = t1ce.view(-1, self.output_dim)
        loss_d = self.cross_entropy[0](
            self.mlp(d_fake), torch.zeros_like(d_fake)) + self.cross_entropy[1](
            self.mlp(d_real), torch.ones_like(d_real))
        loss_g = self.cross_entropy[1](
            self.mlp(d_fake), torch.ones_like(d_fake))
        loss += loss_d + loss_g

        return {
            "loss": loss,
            "acc_result": accumulate_cv_data({
                "output": fake.detach(),
                "gt": t1ce.detach(),
            }, acc_result, config, mode),
            "output": list(torch.split(fake.squeeze().cpu().detach(), 1, dim=0))
        }
