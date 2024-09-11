import torch
import torch.nn as nn
import torch.nn.functional as F


class ElementLoss(nn.Module):
    def __init__(self):
        super(ElementLoss, self).__init__()
        # 6 main elements in the head
        self.elements = nn.parameter.Parameter(torch.randn(6))

    def forward(self, x):
        loss = 1
        for e in self.elements:
            loss *= (x - e)
        return torch.linalg.norm(loss)


class GramLoss(nn.Module):
    def __init__(self):
        super(GramLoss, self).__init__()
        self.mse_criterion = nn.MSELoss()

    def forward(self, features, targets, weights=None):
        if weights is None:
            weights = [1/len(features)] * len(features)

        gram_loss = 0
        for f, t, w in zip(features, targets, weights):
            gram_loss += self.mse_criterion(self.gram(f), self.gram(t)) * w
        return gram_loss

    def gram(self, x):
        b, c, h, w = x.size()
        g = torch.bmm(x.view(b, c, h*w), x.view(b, c, h*w).transpose(1, 2))
        return g.div(h*w)


class SobelLoss(nn.Module):
    def __init__(self):
        super(SobelLoss, self).__init__()
        self.l1_criterion = nn.L1Loss()

    def sobel_x(self, x):
        sobel_kernel = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        return F.conv2d(x, sobel_kernel, padding=1)

    def sobel_y(self, x):
        sobel_kernel = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        return F.conv2d(x, sobel_kernel, padding=1)

    def forward(self, pred, target):
        return self.l1_criterion(self.sobel_x(pred), self.sobel_x(target)) + self.l1_criterion(self.sobel_y(pred), self.sobel_y(target))


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.l1_criterion = nn.L1Loss()
        self.sobel_criterion = SobelLoss()

    def forward(self, x, y):
        return self.l1_criterion(x, y) * 0.3 + self.sobel_criterion(x, y) * 0.7
