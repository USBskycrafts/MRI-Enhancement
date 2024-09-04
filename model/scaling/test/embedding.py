from model.scaling.parts import CrossScaleEmbedding
import unittest
import torch
import torch.nn as nn


class Test(unittest.TestCase):
    def test_shape(self):
        x = torch.Tensor(8, 16, 240, 240)
        model = CrossScaleEmbedding(16, 32)
        y = model(x, nn.ModuleList([]))
        print(y.shape)
        model = CrossScaleEmbedding(16, 32, [4, 8, 16, 32], 4)
        y = model(x, nn.ModuleList([]))
        print(y.shape)


if __name__ == '__main__':
    unittest.main()
