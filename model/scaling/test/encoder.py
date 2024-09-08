import unittest
import torch
import torch.nn as nn
import torch.nn.parameter
from model.scaling.transformer_layer import TransformerLayer


class TestEncoder(unittest.TestCase):
    # def test_encode(self):
    #     # TODO: Implement test for encode method
    #     encoder = TransformerLayer(1, 3)
    #     input = torch.arange(0, 36, dtype=torch.float32).reshape(1, 1, 6, 6)
    #     output = encoder(input)
    #     print(output.shape)

    # def test_padding(self):
    #     encoder = TransformerLayer(1, 3)
    #     input = torch.arange(0, 25, dtype=torch.float32).reshape(1, 1, 5, 5)
    #     output = encoder(input)
    #     print(output.shape)
    #     input = torch.arange(0, 20, dtype=torch.float32).reshape(1, 1, 4, 5)
    #     output = encoder(input)
    #     print(output.shape)
    #     input = torch.arange(0, 20, dtype=torch.float32).reshape(1, 1, 5, 4)
    #     output = encoder(input)
    #     print(output.shape)
    #     encoder = TransformerLayer(1, 7)
    #     input = torch.arange(
    #         0, 27 * 29, dtype=torch.float32).reshape(1, 1, 27, 29)
    #     output = encoder(input)
    #     print(output.shape)

    def test_correct_reshape(self):
        class ByPass(nn.Module):
            def __init__(self):
                super(ByPass, self).__init__()

            def forward(self, x):
                return x

        class MockedTransformer(TransformerLayer):
            def __init__(self, *args, **kwargs):
                super(MockedTransformer, self).__init__(*args, **kwargs)
                self.encoder = ByPass()
                self.position_embedding = torch.nn.parameter.Parameter(
                    torch.zeros_like(self.position_embedding)
                )

            def forward(self, x):
                print(self.groups)
                return super(MockedTransformer, self).forward(x)

        encoder = MockedTransformer(1, 3)
        input = torch.arange(0, 72, dtype=torch.float32).reshape(1, 2, 6, 6)
        # output = encoder(input)
        # assert input.equal(output)
        # print(input, "and ", encoder.groups)
        encoder = MockedTransformer(2, 3, 'long')
        output = encoder(input)
        # assert input.equal(output)
        # print(input, "and", encoder.groups)


if __name__ == '__main__':
    unittest.main()
