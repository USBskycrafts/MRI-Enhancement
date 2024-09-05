import unittest
import torch
import torch.nn as nn
from model.scaling.transformer_layer import TransformerLayer


class TestEncoder(unittest.TestCase):
    def test_encode(self):
        # TODO: Implement test for encode method
        encoder = TransformerLayer(4, 3)


if __name__ == '__main__':
    unittest.main()
