from model.scaling.cross_embedding import CrossScaleEmbedding
import unittest
import torch
import torch.nn as nn


class Test(unittest.TestCase):
    def test_shape(self):
        x = torch.Tensor(8, 16, 240, 240)
        model = CrossScaleEmbedding(16, 512)
        y = model(x)
        print(x.shape, y.shape)
        model = CrossScaleEmbedding(16, 512, reversed=True)
        x = model(y, y)
        print(x.shape, y.shape)
        model = CrossScaleEmbedding(16, 32, [4, 8, 16, 32], 4)
        y = model(x)
        print(x.shape, y.shape)
        model = CrossScaleEmbedding(16, 32, [4, 8, 16, 32], 4, reversed=True)
        x = model(y, y)
        print(x.shape, y.shape)


if __name__ == '__main__':
    unittest.main()
