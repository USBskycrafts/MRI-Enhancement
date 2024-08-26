import unittest
import torch.nn.functional as F
import torch.nn.parameter as P
import torch


class KernelTest(unittest.TestCase):
    def test_kernel(self):
        token = P.Parameter(torch.randn(32, 1, 128, 128))
        kernel = P.Parameter(torch.randn(3, 1, 128, 128))
        result = F.conv2d(token, kernel, stride=1, padding=0).squeeze(2)
        print(result.shape)
