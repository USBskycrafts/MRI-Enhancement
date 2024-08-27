import unittest
import torch
import matplotlib.pyplot as plt
from tools.augmentation_tool import sobel_edge, cutout


class TestAugmentation(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.data = torch.rand((3, 128, 128))

    def test_sobel_edge(self):
        # plt.figure("sobel test")
        plt.subplot(2, 1, 1)
        plt.imshow(torch.permute(self.data, (1, 2, 0)).numpy())
        plt.subplot(2, 1, 2)
        plt.imshow(torch.permute(sobel_edge(
            self.data, None), (1, 2, 0)).numpy())
        # plt.show()
        # plt.savefig("test/tests.png")
        return

    def test_cutout(self):
        # plt.figure("cutout test")
        plt.subplot(2, 1, 1)
        plt.imshow(torch.permute(self.data, (1, 2, 0)).numpy())
        plt.subplot(2, 1, 2)
        plt.imshow(torch.permute(cutout(
            self.data, {"data": '[16, 16]'}), (1, 2, 0)).numpy())
        # plt.show()
        return

    def test_dim(self):
        plt.figure("dim test")
        data = torch.stack([self.data, cutout(
            self.data, {"data": '[16, 16]'}), sobel_edge(self.data, None)], dim=0)
        plt.subplot(2, 1, 1)
        plt.imshow(torch.permute(data[0], (1, 2, 0)).numpy())
        # plt.show()
        plt.savefig("test/tests.png")


if __name__ == "__main__":
    unittest.main()
