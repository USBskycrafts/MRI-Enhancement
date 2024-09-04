import os
import nibabel as nib
from nibabel.nifti1 import load
import torch
import logging
from tools.augmentation_tool import crop_to_list
from itertools import chain
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
        self.input_dim = config.getint("model", "input_dim")
        self.output_dim = config.getint("model", "output_dim")

        t1_list, t2_list, t1ce_list = [], [], []
        self.logger.info(f"""Loading {mode} data from {
                     self.t1_dir} and {self.t1ce_dir} and {self.t2_dir}""")

        for file in sorted(os.listdir(self.t1_dir)):
            if file.endswith(".nii"):
                path = os.path.join(self.t1_dir, file)
                data = load(path).get_fdata()
                tensor = torch.tensor(data)
                tensor = (tensor - tensor.min()) / \
                    (tensor.max() - tensor.min())
                tensor = torch.permute(tensor, (2, 0, 1)).float()

                mid_layer = tensor.shape[0] // 2
                radio = tensor.shape[0] // 4
                tensor = tensor[mid_layer-radio:mid_layer+radio, :, :]
                tensors = list(torch.split(tensor, self.input_dim, dim=0))
                if tensors[-1].shape[0] < self.input_dim:
                    tensors.pop()
                t1_list.extend(tensors)
        # print(len(t1_list))
        # print(t1_list[0].shape)
        if mode == "train":
            t1_list = sum(
                list(map(lambda x: crop_to_list(x, config), t1_list)), [])
        # print(len(t1_list))
        # print(t1_list[0].shape)
        for file in sorted(os.listdir(self.t2_dir)):
            if file.endswith(".nii"):
                path = os.path.join(self.t2_dir, file)
                data = load(path).get_fdata()
                tensor = torch.tensor(data)
                tensor = (tensor - tensor.min()) / \
                    (tensor.max() - tensor.min())
                tensor = torch.permute(tensor, (2, 0, 1)).float()
                mid_layer = tensor.shape[0] // 2
                radio = tensor.shape[0] // 4
                tensor = tensor[mid_layer-radio:mid_layer+radio, :, :]
                tensors = list(torch.split(tensor, self.input_dim, dim=0))
                if tensors[-1].shape[0] < self.input_dim:
                    tensors.pop()
                t2_list.extend(tensors)
        if mode == "train":
            t2_list = sum(
                list(map(lambda x: crop_to_list(x, config), t2_list)), [])
        for file in sorted(os.listdir(self.t1ce_dir)):
            if file.endswith(".nii"):
                path = os.path.join(self.t1ce_dir, file)
                data = load(path).get_fdata()
                tensor = torch.tensor(data)
                tensor = (tensor - tensor.min()) / \
                    (tensor.max() - tensor.min())
                tensor = torch.permute(tensor, (2, 0, 1)).float()
                mid_layer = tensor.shape[0] // 2
                radio = tensor.shape[0] // 4
                tensor = tensor[mid_layer-radio:mid_layer+radio, :, :]
                tensors = list(torch.split(tensor, self.input_dim, dim=0))
                if tensors[-1].shape[0] < self.input_dim:
                    tensors.pop()
                t1ce_list.extend(tensors)
        if mode == "train":
            t1ce_list = sum(
                list(map(lambda x: crop_to_list(x, config), t1ce_list)), [])
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
