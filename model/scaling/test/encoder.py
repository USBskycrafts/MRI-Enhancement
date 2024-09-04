import unittest
import torch
import torch.nn as nn


class TransformerLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, group=3):
        super(TransformerLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.group = group

        self.encoder = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=8)

    def padding(self, x):
        # WARN: the image should be padded to the multiple of group
        pass

    def forward(self, x):
        return self.encoder(x)


class TestEncoder(unittest.TestCase):
    def test_encode(self):
        # TODO: Implement test for encode method
        encoder = TransformerLayer(512, 128, 8)
        x = torch.randn(32, 16, 16, 7, 7, 512)
        tokens = x.reshape(-1, 7 * 7, 512)
        print(tokens.shape)
        y = encoder(tokens)
        y = y.reshape(x.shape)
        print(y.shape)

    def test_reverse(self):
        x = torch.randn(32, 16, 16, 7, 7, 128)
        tokens = x.reshape(-1, 7 * 7, 128)
        assert tokens.reshape(x.shape).equal(x)


if __name__ == '__main__':
    unittest.main()
