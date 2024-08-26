import torch
from typing import List
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor


def render_results(result: List[torch.Tensor], config, *args, **params):
    """Render results from the model

    Args:
        result (List[torch.Tensor]): the shape of the tensor is [1, height, width]
        config (_type_): _description_
    """
    render_path = config.get("test", "render_path")
    for i, res in enumerate(result):
        plt.imsave(f"{render_path}/result_{i}.png",
                   res[0].cpu().numpy(), cmap="gray")
    return


class ResultRenderer:
    def __init__(self, config):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=4)

    def render_results(self, result: List[torch.Tensor], *args, **params):
        self.executor.submit(render_results, result,
                             self.config, *args, **params)
