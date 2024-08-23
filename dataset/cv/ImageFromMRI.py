import os
import nibabel as nib
import torch
import logging
from torch.utils.data import Dataset


class ImageFromMRI(Dataset):
    def __init__(self, config, mode, *args, **params):
        self.config = config
        self.mode = mode
        self.data_list = []
        self.origin_dir = config.get("data", "%s_origin_dir" % mode)
        self.gt_dir = config.get("data", "%s_gt_dir" % mode)
        self.logger = logging.getLogger(__name__)

        origin_list, target_list = [], []
        self.logger.info(f"""Loading {mode} data from {
                     self.origin_dir} and {self.gt_dir}""")
        for file in os.listdir(self.origin_dir):
            if file.endswith(".nii"):
                path = os.path.join(self.origin_dir, file)
                data = nib.load(path).get_fdata()  # type: ignore
                tensor = torch.tensor(data)
                tensor = (tensor - tensor.min()) / \
                    (tensor.max() - tensor.min())
                tensor = torch.permute(tensor, (2, 0, 1)).float()
                tensors = tensor.split(1, dim=0)
                origin_list.extend(tensors)

        for file in os.listdir(self.gt_dir):
            if file.endswith(".nii"):
                path = os.path.join(self.gt_dir, file)
                data = nib.load(path).get_fdata()  # type: ignore
                tensor = torch.tensor(data)
                tensor = (tensor - tensor.min()) / \
                    (tensor.max() - tensor.min())
                tensor = torch.permute(tensor, (2, 0, 1)).float()
                tensors = tensor.split(1, dim=0)
                target_list.extend(tensors)

        assert (len(origin_list) == len(target_list))
        self.logger.info(f"Total {len(origin_list)} {mode} data loaded")
        self.data_list = list(zip(origin_list, target_list))
        self.data_list = list(map(lambda x: {
            "origin": x[0],
            "target": x[1]
        }, self.data_list))

    def __getitem__(self, item):
        return {
            "origin": self.data_list[item]["origin"],
            "target": self.data_list[item]["target"]
        }

    def __len__(self):
        return len(self.data_list)
