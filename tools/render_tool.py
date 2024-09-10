import random
import torch.nn.functional as F
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


def render_results(origin, auxiliary, gt, result: List[torch.Tensor], batch, config, *args, **params):
    """Render results from the model

    Args:
        result (List[torch.Tensor]): the shape of the tensor is [1, height, width]
        config (_type_): _description_
    """
    lock = params["lock"]
    render_path = params["render_path"]
    o, a, t, r = list(zip(origin, auxiliary, gt, result))[len(result) // 2]
    o = torch.permute(o.squeeze(1), (1, 2, 0))
    t = torch.permute(t.squeeze(1), (1, 2, 0))
    r = torch.permute(r.squeeze(1), (1, 2, 0))
    a = torch.permute(a.squeeze(1), (1, 2, 0))
    with lock:
        ax = plt.subplot(1, 4, 1)
        ax.set_title("T1")
        plt.imshow(o.cpu().numpy(), cmap="gray", vmax=1, vmin=0)
        bx = plt.subplot(1, 4, 3)
        bx.set_title("T1CE")
        plt.imshow(t.cpu().numpy(), cmap="gray", vmax=1, vmin=0)
        cx = plt.subplot(1, 4, 4)
        cx.set_title("Result")
        plt.imshow(r.cpu().numpy(), cmap="gray", vmax=1, vmin=0)
        dx = plt.subplot(1, 4, 2)
        dx.set_title("T2")
        plt.imshow(a.cpu().numpy(), cmap="gray", vmax=1, vmin=0)
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
                                        config.get("output", "model_name"),
                                        time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
        if not os.path.exists(self.render_path):
            os.makedirs(self.render_path)
        self.futures = []

    def __del__(self):
        concurrent.futures.wait(self.futures)

    def render_results(self, data: Dict[str, torch.Tensor], result: List[torch.Tensor], batch,  *args, **params):
        def task(data, result):
            t1 = torch.detach(data["t1"].cpu())
            t1ce = torch.detach(data["t1ce"].cpu())
            t2 = torch.detach(data["t2"].cpu())
            t1 = t1.split(1, dim=0)
            t1ce = t1ce.split(1, dim=0)
            t2 = t2.split(1, dim=0)
            render_results(t1, t2, t1ce, result, batch,
                           self.config, *args, lock=self.lock, render_path=self.render_path, **params)
        future = self.executor.submit(task, data, result)
        self.futures.append(future)
        task(data, result)
