import unittest
from tools.output_tool import cv_output_function
from tools.eval_tool import output_value
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

    def test_color(self):
        output_value(1, "train", 3, 10, 0.1, {}, None, None)
        output_value(1, "eval", 3, 10, 0.1, {}, None, None)


if __name__ == '__main__':
    unittest.main()
