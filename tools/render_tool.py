import threading
import torch
from typing import List, Dict
import matplotlib.pyplot as plt
import logging
import os
from concurrent.futures import ThreadPoolExecutor, wait


def render_results(origin, gt, result: List[torch.Tensor], config, batch, *args, **params):
    """Render results from the model

    Args:
        result (List[torch.Tensor]): the shape of the tensor is [1, height, width]
        config (_type_): _description_
    """
    lock = params["lock"]
    render_path = config.get("test", "render_path")
    for i, (o, t, r) in enumerate(zip(origin, gt, result)):
        o = torch.permute(o, (1, 2, 0))
        t = torch.permute(t, (1, 2, 0))
        r = torch.permute(r, (1, 2, 0))
        # with lock:
        plt.subplot(2, 2, 1)
        plt.imshow(o.cpu().numpy(), cmap="gray")
        plt.subplot(2, 2, 2)
        plt.imshow(t.cpu().numpy(), cmap="gray")
        plt.subplot(2, 2, 3)
        plt.imshow(r.cpu().numpy(), cmap="gray")
        plt.savefig(f"{render_path}/{batch}-{i}.png",
                    bbox_inches='tight', dpi=600)
    return


class ResultRenderer:
    def __init__(self, config):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=128)
        self.logger = logging.getLogger(__name__)
        self.lock = threading.Lock()
        self.futures = []
        if not os.path.exists(config.get("test", "render_path")):
            os.makedirs(config.get("test", "render_path"))

    def __del__(self):
        wait(self.futures)

    def render_results(self, data: Dict[str, torch.Tensor], result: List[torch.Tensor], batch, *args, **params):
        def task():
            origin = data["origin"].cpu().detach()
            target = data["target"].cpu().detach()
            origin = origin[:, 0, :, :].split(1, dim=0)
            target = target[:, 0, :, :].split(1, dim=0)
            render_results(origin, target, list(map(lambda x: x.cpu().detach(), result)),
                           self.config, batch, *args, lock=self.lock, **params)
        future = self.executor.submit(task)
        self.futures.append(future)
        # task()
