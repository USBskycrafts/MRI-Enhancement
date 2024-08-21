from formatter.Basic import BasicFormatter
from tools.augmentation_tool import (
    cutout,
    sobel_edge
)
import torch
import logging


class NibabelFormmater(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        logging.info(f"NibabelFormmater: {config}, {mode}")
        self.config = config
        self.mode = mode

    def process(self, data, config, mode, *args, **params):
        origin_list = list(
            map(lambda x: x["origin"], data))
        target_list = list(
            map(lambda x: x["target"], data))
        origin_list = list(map(lambda x: torch.cat(
            [x, sobel_edge(x, config), cutout(x, config)], dim=0), origin_list))
        return {
            "origin": torch.stack(origin_list, dim=0),
            "target": torch.stack(target_list, dim=0)
        }
