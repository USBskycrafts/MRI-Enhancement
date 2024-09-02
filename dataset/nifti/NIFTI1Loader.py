import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import logging
import nibabel as nib


class NIFTI1Loader(Dataset):
    def __init__(self, config, mode, *args, **kwargs):
        super().__init__()
        self.config = config
        self.mode = mode
        self.t1_dir = config.get("data", "%s_t1_dir" % mode)
        self.t2_dir = config.get("data", "%s_t2_dir" % mode)
        self.t1ce_dir = config.get("data", "%s_t1ce_dir" % mode)
        self.logger = logging.getLogger(__name__)
        self.input_dim = config.getint("model", "input_dim")
        self.output_dim = config.getint("model", "output_dim")
        self.data_list = []

        for (T1, T2, T1CE) in zip(sorted(os.listdir(self.t1_dir)),
                                  sorted(os.listdir(self.t2_dir)),
                                  sorted(os.listdir(self.t1ce_dir))):
            if T1.endwith(".nii") and T2.endwith(".nii") and T1CE.endwith(".nii"):
                def load_from_path(dir, path):
                    path = os.path.join(dir, path)
                    image = nib.nifti1.load(path)
                    # transform to standard pytorch tensor
                    tensor = torch.Tensor(image.get_fdata())
                    tensor = tensor.permute(2, 0, 1)
                    # normalize to [0, 1]
                    tensor = (tensor - tensor.min()) / \
                        (tensor.max() - tensor.min())
                    return tensor
                T1, T2, T1CE = map(load_from_path, [
                    self.t1_dir, self.t2_dir, self.t1ce_dir], [T1, T2, T1CE])
                self.data_list.append({
                    "t1", T1,
                    "t2", T2,
                    "t1ce", T1CE
                })
