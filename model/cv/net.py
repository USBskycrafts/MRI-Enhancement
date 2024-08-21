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
        # output should be the dimension of the original data
        self.output_dim = config.getint("model", "output_dim")

        # encoder and decoder are the skeleton of the model
        self.encoder = Encoder(config, gpu_list, *args, **params)
        self.decoder = Decoder(config, gpu_list, *args, **params)

        # the loss for feature tokens
        self.cosine_loss = nn.ModuleList(
            nn.CosineSimilarity(dim=1) for _ in range(self.input_dim * (self.input_dim - 1) // 2))
        # the loss for the output of model
        self.terminal_loss = nn.ModuleList(
            nn.L1Loss() for _ in range(self.output_dim))

    def forward(self, data, config, gpu_list, acc_result, mode):
        x, gt = data["origin"], data["target"]
        loss = 0
        tokens = []
        for i, clip in enumerate(torch.split(x, 1, dim=1)):
            features = self.encoder(clip, config, gpu_list, acc_result, mode)
            y = self.decoder(features, config, gpu_list,
                             acc_result, mode)["logits"]
            loss += self.terminal_loss[i](y, gt)
            tokens.append(features["token"])
        for e1, e2 in itertools.combinations(tokens, 2):
            loss += sum(list(map(lambda x: x(e1, e2), self.cosine_loss)))
        return {
            "loss": loss,
            "acc_result": accumulate_cv_data({
                "output": y,
                "gt": gt,
            }, acc_result, config, mode),
            "output": torch.split(y, 1, dim=0) if mode == "test" else []
        }
