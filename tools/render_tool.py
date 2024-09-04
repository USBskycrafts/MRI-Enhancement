import random
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import threading
import torch
from typing import List, Dict
import matplotlib.pyplot as plt
plt.switch_backend("agg")


def render_results(origin, gt, result: List[torch.Tensor], batch, config, *args, **params):
    """Render results from the model

    Args:
        result (List[torch.Tensor]): the shape of the tensor is [1, height, width]
        config (_type_): _description_
    """
    lock = params["lock"]
    render_path = params["render_path"]
    o, t, r = list(zip(origin, gt, result))[len(result) // 2]
    o = torch.permute(o, (1, 2, 0))
    t = torch.permute(t, (1, 2, 0))
    r = torch.permute(r, (1, 2, 0))
    with lock:
        ax = plt.subplot(1, 3, 1)
        ax.set_title("T1")
        plt.imshow(o.cpu().numpy(), cmap="gray")
        bx = plt.subplot(1, 3, 2)
        bx.set_title("T1CE")
        plt.imshow(t.cpu().numpy(), cmap="gray")
        cx = plt.subplot(1, 3, 3)
        cx.set_title("Result")
        plt.imshow(r.cpu().numpy(), cmap="gray")
        plt.savefig(f"{render_path}/{batch}.png",
                    dpi=600, bbox_inches='tight')
    return


class ResultRenderer:
    def __init__(self, config):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=32)
        self.logger = logging.getLogger(__name__)
        self.lock = threading.Lock()
        self.render_path = os.path.join(config.get("test", "render_path"),
                                        time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
        if not os.path.exists(self.render_path):
            os.makedirs(self.render_path)
        self.futures = []

    def __del__(self):
        concurrent.futures.wait(self.futures)

    def render_results(self, data: Dict[str, torch.Tensor], result: List[torch.Tensor], batch,  *args, **params):
        def task(data, result):
            t1 = torch.detach(data["t1"].cpu()).squeeze()
            t1ce = torch.detach(data["t1ce"].cpu()).squeeze()
            t1 = t1.split(1, dim=0)
            t1ce = t1ce.split(1, dim=0)
            render_results(t1, t1ce, result, batch,
                           self.config, *args, lock=self.lock, render_path=self.render_path, **params)
        future = self.executor.submit(task, data, result)
        self.futures.append(future)
        task(data, result)
