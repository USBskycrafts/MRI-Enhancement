import torch
import torch.nn as nn
import unittest
from typing import List, Union


class CrossScaleEmbedding(nn.Module):
    # Should be used in pairs
    def __init__(self, input_dim: int, output_dim: int,
                 kernel_size: List[int] = [2, 4],
                 stride: int = 2, reversed: bool = False):
        super(CrossScaleEmbedding, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = [k for k in sorted(kernel_size)]
        self.stride = stride
        self.reversed = reversed

        self.convs = nn.ModuleList()
        token_size = self.token_size(self.kernel_size, output_dim)
        self.dim_list = token_size
        if not reversed:
            for i, k in enumerate(self.kernel_size):
                self.convs.append(
                    nn.Conv2d(input_dim, token_size[i],
                              kernel_size=k, stride=stride, padding=self.padding_size(k, stride)))
        else:
            for i, k in enumerate(self.kernel_size):
                self.convs.append(
                    # Warning: may cause error if H and W are not even
                    nn.ConvTranspose2d(2 * token_size[i], input_dim,
                                       kernel_size=k, stride=stride, padding=self.padding_size(k, stride)))

    def token_size(self, kernel_size, output_dim) -> List[int]:
        token_dim = []
        for i in range(1, len(kernel_size)):
            token_dim.append(output_dim // (2**i))
            if i == len(kernel_size) - 1:
                # the largest token dim should equals to the
                # secondary largest token dim
                token_dim.append(output_dim // (2**i))
        return token_dim

    def padding_size(self, kernel_size, stride) -> int:
        """Calculate padding size for convolution

        Args:
            kernel_size (_type_): _description_
            stride (_type_): _description_

        Returns:
            _type_: _description_
        while dilation=1,
        y.shape = (x.shape + 2 * p.shape - k.shape) // stride + 1
        if we want y.shape = x.shape // stride
        then we get this function
        """
        if (kernel_size - stride) % 2 == True:
            return (kernel_size - stride) // 2
        else:
            return (kernel_size - stride + 1) // 2

    def forward(self, x, y=Union[None, torch.Tensor]):
        if not self.reversed:
            # from [B, C, H, W] to [B, H // stride, W // stride, C * stride]
            tokens = torch.cat([conv(x)
                                for conv in self.convs], dim=1)
            # a recursion to the deep layers
            return tokens
        else:
            assert isinstance(y, torch.Tensor)
            assert x.shape == y.shape
            features = torch.zeros(
                x.shape[0], x.shape[1] * 2, x.shape[2], x.shape[3])
            features[:, ::2, :, :] = x
            features[:, 1::2, :, :] = y
            offset = 0
            output = torch.zeros_like(x)
            for i, d in enumerate(self.dim_list):
                output += self.convs[i](features[:,
                                        offset:offset + 2 * d, :, :])
            assert output.shape == x.shape
            return output
