import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerLayer(nn.Module):
    def __init__(self, input_dim: int, group=3):
        super(TransformerLayer, self).__init__()
        self.input_dim = input_dim
        self.group = group

        self.position_embedding = nn.parameter.Parameter(torch.randn(
            (1, group * group, input_dim)
        ))
        self.encoder = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=1,
        )

    def padding(self, x):
        # WARN: the image should be padded to the multiple of group

        # first, calculate the size of the padded image
        x_diff = (x.shape[2] + self.group -
                  1) // self.group * self.group - x.shape[2]
        y_diff = (x.shape[3] + self.group -
                  1) // self.group * self.group - x.shape[3]
        assert x_diff >= 0 and y_diff >= 0, "the image should be padded to the multiple of group"
        padding_size = (
            y_diff // 2,
            y_diff // 2 + (1 if y_diff % 2 == 1 else 0),
            x_diff // 2,
            x_diff // 2 + (1 if x_diff % 2 == 1 else 0),
        )
        x = F.pad(x, padding_size, mode='reflect')
        assert x.shape[2] % self.group == 0, f"{x.shape}"
        assert x.shape[3] % self.group == 0, f"{x.shape}"
        return x

    def forward(self, x):
        x = self.padding(x)
        group_list = torch.split(x, self.group, dim=2)
        group_list = sum(map(lambda e: torch.split(
            e, self.group, dim=3), group_list), ())
        # now the groups' shape is [B * H * W // (G * G), C, G, G]
        groups = torch.cat(group_list, dim=0)
        # then we transform the groups to the standard tokens
        # [B * H * W // (G * G), G * G, C]
        groups = groups.flatten(2, 3).permute(0, 2, 1)
        # groups with position embedding
        groups += self.position_embedding
        groups = self.encoder(groups)
        # then we transform the tokens back to the groups

        # [B, H // G, W // G, G, G, C]
        groups = groups.reshape(-1,
                                x.shape[2] // self.group,
                                x.shape[3] // self.group,
                                self.group, self.group, self.input_dim)

        groups = groups.permute(
            0, 1, 3, 2, 4, 5).reshape(-1, x.shape[2], x.shape[3], self.input_dim)
        groups = groups.permute(0, 3, 1, 2)
        assert groups.shape == x.shape
        return groups
