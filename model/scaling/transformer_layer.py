import torch.nn as nn


class TransformerLayer(nn.Module):
    def __init__(self, input_dim: int, group=3):
        super(TransformerLayer, self).__init__()
        self.input_dim = input_dim
        self.group = group

    def padding(self, x):
        # WARN: the image should be padded to the multiple of group
        pass

    def forward(self, x):
        pass
