import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_loss = torch.nn.L1Loss()
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, input, mask, output, target):
        loss1 = self.mask_loss(F.relu(target - input), mask) * 0.1
        loss2 = self.l1_loss(output, target)
        return loss1 + loss2
