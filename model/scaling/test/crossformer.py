import torch
import torch.nn as nn
import unittest
from thop import profile
from model.scaling.crossformer import CrossformerUnit, CrossScaleParams, CrossformerParams
from .utils import splitter, timer


class Crossformer(nn.Module):
    def __init__(self):
        super(Crossformer, self).__init__()

        self.cross_params = [
            CrossScaleParams(
                input_dim=1,
                output_dim=64,
                kernel_size=[4, 8, 16, 32],
                stride=4),
            CrossScaleParams(
                input_dim=64,
                output_dim=128,
                kernel_size=[2, 4],
                stride=2),
            CrossScaleParams(
                input_dim=128,
                output_dim=256,
                kernel_size=[2, 4],
                stride=2),
            CrossScaleParams(
                input_dim=256,
                output_dim=512,
                kernel_size=[2, 4],
                stride=2),
        ]
        self.transformer_params = [
            CrossformerParams(
                input_dim=64,
                group=7,
                n_layer=1
            ),
            CrossformerParams(
                input_dim=128,
                group=7,
                n_layer=1
            ),
            CrossformerParams(
                input_dim=256,
                group=7,
                n_layer=8
            ),
            CrossformerParams(
                input_dim=512,
                group=7,
                n_layer=6
            )
        ]

        self.transformer_units = nn.ModuleList([])
        for cross_param, transformer_param in zip(self.cross_params, self.transformer_params):
            self.transformer_units.append(
                CrossformerUnit(cross_param, transformer_param))

    def forward(self, x):
        first, rest = self.transformer_units[0], list(
            self.transformer_units)[1:]
        return first(x, rest)


class TestCrossformer(unittest.TestCase):
    @splitter
    @timer
    def test_regular_shape(self):
        self.crossformer = Crossformer()
        x = torch.randn(1, 1, 224, 224)
        y = self.crossformer(x)
        print(x.shape, y.shape)

    @splitter
    @timer
    def test_irregular_shape(self):
        self.crossformer = Crossformer()
        x = torch.randn(1, 1, 251, 211)
        y = self.crossformer(x)
        print(x.shape, y.shape)

    @splitter
    @timer
    def test_backward(self):
        self.crossformer = Crossformer()
        x = torch.randn(1, 1, 240, 240)
        y = self.crossformer(x)
        print(x.shape, y.shape)
        loss = torch.sum(y - x)
        loss.backward()

    @splitter
    @timer
    def test_model_size(self):
        self.crossformer = Crossformer()
        x = torch.randn(1, 1, 240, 240)
        flops, *params = profile(self.crossformer, inputs=(x,))
        if (len(params) == 1):
            print(f"{flops/1e9} GFLOPs, {params[0]/1e6} MParams")


if __name__ == '__main__':
    unittest.main()
