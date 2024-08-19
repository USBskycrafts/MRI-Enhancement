from formatter.Basic import BasicFormatter
import torch
import logging


class NibabelFormmater(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        logging.info(f"NibabelFormmater: {config}, {mode}")
        self.config = config
        self.mode = mode

    def process(self, data, config, mode, *args, **params):
        norm_max = config.getint("data", "norm_max")
        origin_list = list(
            map(lambda x: x["origin"], data))
        target_list = list(
            map(lambda x: x["target"], data))
        return {
            "origin": torch.stack(origin_list, dim=0),
            "target": torch.stack(target_list, dim=0)
        }
