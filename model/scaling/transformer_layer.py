import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerLayer(nn.Module):
    def __init__(self, input_dim: int, group=3, attention_type="long"):
        super(TransformerLayer, self).__init__()
        self.input_dim = input_dim
        self.group = group
        self.attention_type = attention_type

        self.position_embedding = nn.parameter.Parameter(torch.randn(
            (1, group * group, input_dim)
        ))
        self.encoder = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=1,
        )
        self.groups = 0

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
        bs, c, h, w = x.shape
        if self.attention_type == "local":
            groups = F.unfold(x, kernel_size=(self.group, self.group),
                              stride=(self.group, self.group))
            groups = groups.reshape(bs, c, self.group * self.group, -1)
            groups = groups.permute(0, 3, 2, 1)
            groups = groups.reshape(-1, self.group * self.group, c)
            groups += self.position_embedding
            groups = self.encoder(groups)
            groups = groups.reshape(bs, -1, self.group * self.group, c)
            groups = groups.permute(0, 3, 2, 1)
            groups = groups.reshape(bs, c * self.group * self.group, -1)
            y = F.fold(groups, output_size=(h, w),
                       kernel_size=(self.group, self.group),
                       stride=(self.group, self.group))
            assert y.shape == x.shape
            return y
        elif self.attention_type == "long":
            stride = (h // self.group, w // self.group)
            # project the groups
            groups = F.unfold(x, kernel_size=stride,
                              stride=stride)
            groups = groups.reshape(
                bs, c, stride[0] * stride[1], self.group * self.group)
            groups = groups.permute(0, 2, 3, 1)
            groups = groups.reshape(-1, self.group * self.group, c)
            groups += self.position_embedding
            groups = self.encoder(groups)
            groups = groups.reshape(bs, stride[0] * stride[1],
                                    self.group * self.group, c)
            groups = groups.permute(0, 3, 1, 2)
            groups = groups.reshape(bs, c * stride[0] * stride[1],
                                    self.group * self.group)
            y = F.fold(groups, output_size=(h, w), kernel_size=stride,
                       stride=stride)
            assert y.shape == x.shape
            return y
        else:
            raise NotImplementedError("please check the attention type")
