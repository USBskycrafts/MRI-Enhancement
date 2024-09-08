import unittest
import torch
from model.scaling.transformer_layer import TransformerPack


class TestTransformerPack(unittest.TestCase):
    def test_transformer_pack(self):
        model = TransformerPack(64, 7, 8)
        x = torch.randn(16, 64, 48, 48)
        y = model(x)
        print(x.shape, y.shape)

        x = torch.randn(16, 64, 48, 48)
        y = model(x)
        print(x.shape, y.shape)
