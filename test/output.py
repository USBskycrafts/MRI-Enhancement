import unittest
from tools.output_tool import cv_output_function
import random


class OutputTest(unittest.TestCase):
    def test_output(self):
        psnr = [random.uniform(30, 40) for _ in range(30)]
        ssim = [random.uniform(0.8, 1.0) for _ in range(30)]
        print(cv_output_function({
            'psnr': psnr,
            'ssim': ssim
        }, None))
        return


if __name__ == '__main__':
    unittest.main()
