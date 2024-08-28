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
        t1_list = list(
            map(lambda x: x["t1"], data))
        t2_list = list(
            map(lambda x: x["t2"], data)
        )
        t1ce_list = list(
            map(lambda x: x["t1ce"], data))
        return {
            "t1": torch.stack(t1_list, dim=0),
            "t2": torch.stack(t2_list, dim=0),
            "t1ce": torch.stack(t1ce_list, dim=0)
        }
