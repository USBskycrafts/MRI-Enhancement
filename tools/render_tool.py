import threading
import torch
from typing import List, Dict
import matplotlib.pyplot as plt
import logging
import os
from concurrent.futures import ThreadPoolExecutor


def render_results(origin, gt, result: List[torch.Tensor], config, *args, **params):
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
        lock.acquire()
        plt.subplot(2, 2, 1)
        plt.imshow(o.cpu().numpy(), cmap="gray")
        plt.subplot(2, 2, 2)
        plt.imshow(t.cpu().numpy(), cmap="gray")
        plt.subplot(2, 2, 3)
        plt.imshow(r.cpu().numpy(), cmap="gray")
        plt.savefig(f"{render_path}/{i}.png", bbox_inches='tight', dpi=600)
        lock.release()
    return


class ResultRenderer:
    def __init__(self, config):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=32)
        self.logger = logging.getLogger(__name__)
        self.lock = threading.Lock()
        if not os.path.exists(config.get("test", "render_path")):
            os.makedirs(config.get("test", "render_path"))

    def __del__(self):
        self.executor.shutdown(wait=True)

    def render_results(self, data: Dict[str, torch.Tensor], result: List[torch.Tensor], *args, **params):
        def task():
            origin = data["origin"].cpu().detach()
            target = data["target"].cpu().detach()
            origin = origin[:, 0, :, :].split(1, dim=0)
            target = target[:, 0, :, :].split(1, dim=0)
            render_results(origin, target, list(map(lambda x: x.cpu().detach(), result)),
                           self.config, *args, lock=self.lock, **params)
        # self.executor.submit(task)
        task()
