import torch
import torch.nn as nn
from .transformer_layer import TransformerPack
from .cross_embedding import CrossScaleEmbedding
from typing import Dict, Any, List


class CrossScaleParams:
    def __init__(self, input_dim: int,
                 output_dim: int,
                 kernel_size: List[int],
                 stride: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride = stride

    def keys(self):
        return ['input_dim', 'output_dim', 'kernel_size', 'stride']

    def __getitem__(self, key):
        return getattr(self, key)


class TransformerParams:
    def __init__(self, input_dim: int,
                 group: int,
                 n_layer: int):
        self.input_dim = input_dim
        self.group = group
        self.n_layer = n_layer

    def keys(self):
        return ['input_dim', 'group', 'n_layer']

    def __getitem__(self, key):
        return getattr(self, key)


class TransformerUnit(nn.Module):
    def __init__(self, cross_scale_params: CrossScaleParams, transformer_params: TransformerParams):
        super(TransformerUnit, self).__init__()
        self.encoder_embedding = CrossScaleEmbedding(
            **dict(cross_scale_params))
        self.encoder_layers = TransformerPack(**dict(transformer_params))
        self.decoder_embedding = CrossScaleEmbedding(
            **cross_scale_params, reversed=True)
        transformer_params.input_dim = cross_scale_params.input_dim
        self.decoder_layers = TransformerPack(**dict(transformer_params))

    def forward(self, x, next: List[nn.Module]):
        h = self.encoder_embedding(x)
        h = self.encoder_layers(h)

        if len(next) > 0:
            first, rest = next[0], next[1:]
            h_next = first(h, rest)
        else:
            h_next = h

        h = self.decoder_embedding(h_next, h, x.shape)
        y = self.decoder_layers(h)
        return y
