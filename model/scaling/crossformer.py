import torch
import torch.nn as nn
from .crossformer_layer import CrossformerPack
from .cross_embedding import CrossScaleEmbedding
from typing import Dict, Any, List, Tuple


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


class CrossformerParams:
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


class CrossformerUnit(nn.Module):
    def __init__(self, cross_scale_params: CrossScaleParams, transformer_params: CrossformerParams):
        super(CrossformerUnit, self).__init__()
        self.encoder_embedding = CrossScaleEmbedding(
            **dict(cross_scale_params))
        self.encoder_layers = CrossformerPack(**dict(transformer_params))
        self.decoder_embedding = CrossScaleEmbedding(
            **cross_scale_params, reversed=True)
        transformer_params.input_dim = cross_scale_params.input_dim
        self.decoder_layers = CrossformerPack(**dict(transformer_params))

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


class CrossformerPairBuilder:
    def __init__(self, cross_scale_params: CrossScaleParams, transformer_params: CrossformerParams):
        self.encoder = nn.Sequential(
            CrossScaleEmbedding(
                **dict(cross_scale_params)),
            CrossformerPack(**dict(transformer_params))
        )
        transformer_params.input_dim = cross_scale_params.input_dim

        class SequentialAdapter(nn.Module):
            def __init__(self):
                super(SequentialAdapter, self).__init__()
                self.embedding = CrossScaleEmbedding(
                    **cross_scale_params, reversed=True)

            def forward(self, input):
                x, h, shape = input
                return self.embedding(x, h, shape)

        self.decoder = nn.Sequential(
            SequentialAdapter(),
            CrossformerPack(**dict(transformer_params))
        )

    def build(self) -> Tuple[nn.Module, nn.Module]:
        return self.encoder, self.decoder
