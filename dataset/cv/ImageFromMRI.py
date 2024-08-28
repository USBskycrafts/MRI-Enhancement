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
        self.t1_dir = config.get("data", "%s_t1_dir" % mode)
        self.t2_dir = config.get("data", "%s_t2_dir" % mode)
        self.t1ce_dir = config.get("data", "%s_t1ce_dir" % mode)
        self.logger = logging.getLogger(__name__)

        t1_list, t2_list, t1ce_list = [], [], []
        self.logger.info(f"""Loading {mode} data from {
                     self.t1_dir} and {self.t1ce_dir} and {self.t2_dir}""")
        for file in sorted(os.listdir(self.t1_dir)):
            if file.endswith(".nii"):
                path = os.path.join(self.t1_dir, file)
                data = nib.load(path).get_fdata()  # type: ignore
                tensor = torch.tensor(data)
                tensor = (tensor - tensor.min()) / \
                    (tensor.max() - tensor.min())
                tensor = torch.permute(tensor, (2, 0, 1)).float()
                tensors = tensor.split(1, dim=0)
                t1_list.extend(tensors)

        for file in sorted(os.listdir(self.t2_dir)):
            if file.endswith(".nii"):
                path = os.path.join(self.t2_dir, file)
                data = nib.load(path).get_fdata()  # type: ignore
                tensor = torch.tensor(data)
                tensor = (tensor - tensor.min()) / \
                    (tensor.max() - tensor.min())
                tensor = torch.permute(tensor, (2, 0, 1)).float()
                tensors = tensor.split(1, dim=0)
                t2_list.extend(tensors)

        for file in sorted(os.listdir(self.t1ce_dir)):
            if file.endswith(".nii"):
                path = os.path.join(self.t1ce_dir, file)
                data = nib.load(path).get_fdata()  # type: ignore
                tensor = torch.tensor(data)
                tensor = torch.permute(tensor, (2, 0, 1)).float()
                tensors = tensor.split(1, dim=0)
                t1ce_list.extend(tensors)

        assert (len(t1_list) == len(t1ce_list)
                and len(t1_list) == len(t2_list))
        self.logger.info(f"Total {len(t1_list)} {mode} data loaded")
        self.data_list = list(zip(t1_list, t2_list, t1ce_list))
        self.data_list = list(map(lambda x: {
            "t1": x[0],
            "t2": x[1],
            "t1ce": x[2]
        }, self.data_list))

    def __getitem__(self, item):
        return {
            "t1": self.data_list[item]["t1"],
            "t2": self.data_list[item]["t2"],
            "t1ce": self.data_list[item]["t1ce"],
        }

    def __len__(self):
        return len(self.data_list)
