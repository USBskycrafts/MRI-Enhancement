import logging
from typing import Dict, List
import torch
from formatter.Basic import BasicFormatter


class NIFTI1Formatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)
        self.config = config
        self.mode = mode
        self.logger = logging.getLogger(__name__)

    def process(self, data: List[Dict[str, torch.Tensor]], config, mode, *args, **params):
        t1_list = [t1 for t1 in map(lambda x: x['t1'], data)]
        t2_list = [t2 for t2 in map(lambda x: x['t2'], data)]
        t1ce_list = [t1ce for t1ce in map(lambda x: x['t1ce'], data)]
        # t1_list, t2_list, t1ce_list = map(lambda x: self.data_process(
        #     x, config, mode, *args, **params), [t1_list, t2_list, t1ce_list])
        return {
            't1': torch.stack(t1_list, dim=0),
            't2': torch.stack(t2_list, dim=0),
            't1ce': torch.stack(t1ce_list, dim=0)
        }

    # def data_process(self, data: List[torch.Tensor], config, mode, *args, **params):
    #     # TODO: 1. crop from 155 to 16 in channels(using the middle half channels)
    #     #       2. resize the shape from 240x240 to 64x64 for training
    #     data_list = sum([self.channel_process(
    #         x, config, mode, *args, **params) for x in data], [])
    #     data_list = list(map(lambda x: self.size_process(
    #         x, config, mode, *args, **params), data_list))
    #     return data_list

    # def channel_process(self, data: torch.Tensor, config, mode, *args, **params) -> List[torch.Tensor]:
    #     n_channels = data.shape[0]
    #     start = n_channels // 2 - n_channels // 4
    #     end = n_channels // 2 + n_channels // 4
    #     data = data[start:end, :, :]

    #     input_dim = config.getint("model", "input_dim")
    #     data_list = list(data.split(input_dim, dim=0))
    #     if data_list[-1].shape[0] < input_dim:
    #         data_list.pop()
    #     return data_list

    # def size_process(self, data: torch.Tensor, config, mode, *args, **params) -> torch.Tensor:
    #     if mode == 'train':
    #         return data[:, 30:220, 50:200]
    #     else:
    #         return data
