import unittest
import torch


class RenderTest(unittest.TestCase):
    def test_render(self):
        image = torch.Tensor(32, 3, 32, 32)
        print(image[:, 0, :, :].shape)
        image = image[:, 0, :, :].split(1, dim=0)
        print(image[0].shape)
