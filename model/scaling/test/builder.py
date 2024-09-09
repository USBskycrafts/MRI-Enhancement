import torch
import unittest
from model.scaling.crossformer import CrossformerPairBuilder, CrossScaleParams, CrossformerParams
from .utils import timer, splitter


class TestCrossformerPairBuilder(unittest.TestCase):
    @splitter
    @timer
    def test_build(self):
        scale_param = CrossScaleParams(
            input_dim=3,
            output_dim=64,
            kernel_size=[4, 8, 16, 32],
            stride=4)
        crossformer_param = CrossformerParams(
            input_dim=64,
            group=7,
            n_layer=1
        )
        builder = CrossformerPairBuilder(scale_param, crossformer_param)
        encoder, decoder = builder.build()
        x = torch.randn(1, 3, 240, 240)
        h = encoder(x)
        y = decoder([h, h, x.shape])
        print(x.shape, y.shape)


if __name__ == '__main__':
    unittest.main()
