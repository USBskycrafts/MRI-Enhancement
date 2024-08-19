import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss1 = torch.nn.L1Loss()
        self.loss2 = torch.nn.L1Loss()

    def forward(self, input, mask, output, target):
        loss1 = self.loss1(F.relu(target - input), mask)
        loss2 = self.loss2(output, target)
        return loss1 + loss2
