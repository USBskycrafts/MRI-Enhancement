import torch
import torch.nn as nn
import unittest

from model.cv.net import ProposedModel
from model.scaling.test.utils import splitter, timer
from thop import profile


class TestSymbiosis(unittest.TestCase):

    @splitter
    @timer
    def test_symbiosis_size(self):

        class MockedConfig():
            def __init__(self, input_dim, output_dim):
                self.model = 1
                self.mode = "train"
                self.output = "{}"

            def getint(self, key, default=0):
                return getattr(self, key)

            def get(self, key, default=None):
                return getattr(self, key)

        x = {
            "t1": torch.randn((1, 1, 240, 240)),
            "t2": torch.randn((1, 1, 240, 240)),
            "t1ce": torch.randn((1, 1, 240, 240)),
        }
        model = ProposedModel(MockedConfig(1, 1), [])
        flops, *params = profile(model, inputs=(x,
                                 MockedConfig(1, 1), [], {}, 'train', None))
        if (len(params) == 1):
            print(f"flops: {flops/1e9} GFLOPs, params: {params[0]/1e6} M")
