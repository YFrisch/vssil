import unittest

import torch

from pytorch_ssim import SSIM


class SSIM_Unittest(unittest.TestCase):

    """ Unit-tests for the Structural Similarity Image Metric (SSIM)

        https://github.com/Po-Hsun-Su/pytorch-ssim
    """

    def setUp(self) -> None:
        self.ssim_module = SSIM()

    def test_same_image(self):
        """ Equal images should have a metric of 1. """
        img1 = torch.ones(size=(1, 1, 5, 5))
        img1[0, 0] = torch.eye(5, 5)
        loss = self.ssim_module(img1, img1)
        self.assertEqual(loss, 1)

    def test_small_difference(self):
        """ Images that don't differ much should have a higher metric score. """
        img1 = torch.ones(size=(1, 1, 5, 5))
        img1[0, 0] = torch.eye(5, 5)
        img2 = torch.ones(size=(1, 1, 5, 5))
        img2[0, 0] = torch.eye(5, 5)
        img2[0, 0, 2, 2] = 0.5
        loss = self.ssim_module(img1, img2)
        self.assertTrue(loss >= 0.75)

    def test_random_image(self):
        """ A random image should get a very low metric score. """
        img1 = torch.ones(size=(1, 1, 5, 5))
        img1[0, 0] = torch.eye(5, 5)
        img2 = torch.randn_like(img1).abs().clip(0.0, 1.0)
        loss = self.ssim_module(img1, img2)
        self.assertTrue(loss <= 0.25)


if __name__ == "__main__":
    unittest.main()
