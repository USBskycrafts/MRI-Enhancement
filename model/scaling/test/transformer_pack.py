import unittest
import torch
from model.scaling.transformer_layer import TransformerPack
from .utils import splitter, timer


class TestTransformerPack(unittest.TestCase):
    @splitter
    @timer
    def test_large_patches(self):
        model = TransformerPack(64, 7, 8)
        x = torch.randn(16, 64, 48, 48)
        y = model(x)
        print(x.shape, y.shape)

    @splitter
    @timer
    def test_deep_dim1(self):
        model = TransformerPack(512, 7, 8)
        x = torch.randn(16, 512, 7, 7)
        y = model(x)
        print(x.shape, y.shape)

    @splitter
    @timer
    def test_deep_dim2(self):
        model = TransformerPack(512, 7, 8)
        x = torch.randn(16, 512, 14, 14)
        y = model(x)
        print(x.shape, y.shape)

    @splitter
    @timer
    def test_irregular_shape(self):
        model = TransformerPack(512, 7, 8)
        x = torch.randn(8, 512, 29, 13)
        y = model(x)
        print(x.shape, y.shape)
