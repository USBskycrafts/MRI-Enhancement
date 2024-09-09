import unittest
import torch
import torch.nn as nn
import torch.nn.parameter
from model.scaling.crossformer_layer import CrossformerLayer
from .utils import timer, splitter


class TestCrossformerLayer(unittest.TestCase):
    @splitter
    def test_encode(self):
        # TODO: Implement test for encode method
        encoder = CrossformerLayer(1, 3)
        input = torch.arange(0, 36, dtype=torch.float32).reshape(1, 1, 6, 6)
        output = encoder(input)
        print(output.shape)

    @splitter
    def test_padding(self):
        encoder = CrossformerLayer(1, 3)
        input = torch.arange(0, 25, dtype=torch.float32).reshape(1, 1, 5, 5)
        output = encoder(input)
        print(output.shape)
        input = torch.arange(0, 20, dtype=torch.float32).reshape(1, 1, 4, 5)
        output = encoder(input)
        print(output.shape)
        input = torch.arange(0, 20, dtype=torch.float32).reshape(1, 1, 5, 4)
        output = encoder(input)
        print(output.shape)
        encoder = CrossformerLayer(1, 7, attention_type='local')
        input = torch.arange(
            0, 27 * 29, dtype=torch.float32).reshape(1, 1, 27, 29)
        output = encoder(input)
        print(output.shape)

    @splitter
    def test_correct_reshape(self):
        class ByPass(nn.Module):
            def __init__(self):
                super(ByPass, self).__init__()

            def forward(self, x):
                return x

        class MockedCrossformer(CrossformerLayer):
            def __init__(self, *args, **kwargs):
                super(MockedCrossformer, self).__init__(*args, **kwargs)
                self.encoder = ByPass()
                self.position_embedding = torch.nn.parameter.Parameter(
                    torch.zeros_like(self.position_embedding)
                )

            def forward(self, x):
                return super(MockedCrossformer, self).forward(x)

        input = torch.arange(
            0, 2 * 2 * 17 * 23, dtype=torch.float32).reshape(2, 2, 17, 23)
        encoder = MockedCrossformer(1, 3, 'local')
        output = encoder(input)
        assert input.equal(
            output), f'output is not equal to input: {output}, {input}'
        encoder = MockedCrossformer(2, 3, 'long')
        output = encoder(input)
        assert input.equal(
            output), f'output is not equal to input: {output}, {input}'
        input = torch.arange(
            0, 2 * 2 * 15 * 15, dtype=torch.float32).reshape(2, 2, 15, 15)
        output = encoder(input)
        assert input.equal(
            output), f'output is not equal to input: {output}, {input}'


if __name__ == '__main__':
    unittest.main()
