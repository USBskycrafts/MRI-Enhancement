import torch
import random
import json
from skimage.filters import sobel
from typing import List


def cutout(data: torch.Tensor, config) -> torch.Tensor:
    mask_size: List[int] = json.loads(config.get("data", "mask_size"))
    img_x, img_y = data.shape[1], data.shape[2]
    start_x, start_y = random.randint(
        0, img_x - mask_size[0]), random.randint(0, img_y - mask_size[1])
    # mask as black
    cut = data.clone().detach()
    cut[:, start_x: start_x + mask_size[0],
        start_y: start_y + mask_size[1]] = 0
    return cut


def sobel_edge(data: torch.Tensor, config) -> torch.Tensor:
    data = data.permute((1, 2, 0))
    data = data.cpu().numpy()
    nparray = sobel(data)
    data = torch.from_numpy(nparray)
    data = data.permute((2, 0, 1))
    return data


def crop_to_list(data: torch.Tensor, config) -> List[torch.Tensor]:
    patches = []
    crop_size = 48
    x, y = data.shape[1], data.shape[2]
    for _x in range(crop_size, x, crop_size):
        for _y in range(crop_size, y, crop_size):
            patches.append(data[:, _x:_x+32, _y:_y+32])
    return patches
