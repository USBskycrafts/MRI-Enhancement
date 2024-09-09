import torch
import torch.nn as nn
from .transformer_layer import TransformerPack
from .cross_embedding import CrossScaleEmbedding
from typing import Dict, Any, List


class TransformerUnit(nn.Module):
    def __init__(self, cross_scale_params: Dict[str, Any], transformer_params: Dict[str, Any]):
        super(TransformerUnit, self).__init__()
        self.encoder_embedding = CrossScaleEmbedding(**cross_scale_params)
        self.encoder_layers = TransformerPack(**transformer_params)
        self.decoder_embedding = CrossScaleEmbedding(
            **cross_scale_params, reversed=True)
        self.decoder_layers = TransformerPack(**transformer_params)

    def forward(self, x, next: List[nn.Module]):
        h = self.encoder_embedding(x)
        h = self.encoder_layers(h)

        first, rest = next[0], next[1:]
        if first is not None:
            h_next = first(h, rest)
        else:
            h_next = h

        h = self.decoder_embedding(h_next, h, x.shape)
        y = self.decoder_layers(h)
        return y
