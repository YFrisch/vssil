import yaml
import torch
import unittest

from src.models.transporter import Transporter, \
    TransporterKeypointer, TransporterEncoder, TransporterDecoder

class TransporterUnitTest(unittest.TestCase):

    def setUp(self) -> None:

        with open('/home/yannik/vssil/src/configs/transporter.yml') as conf_file:
            self.config = yaml.safe_load(conf_file)

        self.batch_size = 8
        self.img_channels = self.config['model']['num_img_channels']
        self.num_features = 32
        self.num_keypoints = 1



    def test_encoder(self):
        out_channels = 128
        encoder = TransporterEncoder(self.config)
        sample_image = torch.randn(self.batch_size, self.img_channels, 32, 32)
        feature_maps = encoder(sample_image)
        self.assertEqual(feature_maps.shape, (self.batch_size, out_channels, 8, 8))

    def test_keypointnet(self):
        keypointer = TransporterKeypointer(self.config)
        sample_image = torch.randn(self.batch_size, self.img_channels, 32, 32)
        heatmaps = keypointer(sample_image)
        self.assertEqual(heatmaps.shape, (self.batch_size, self.img_channels, 8, 8))

    def test_decoder(self):
        refine_net = TransporterDecoder(self.config)
        sample_image = torch.randn(self.batch_size, 128, 8, 8)
        heatmaps = refine_net(sample_image)
        self.assertEqual(heatmaps.shape, (self.batch_size, self.img_channels, 32, 32))

    def test_gaussian_maps(self):
        self.num_keypoints = 5
        transporter = Transporter(self.config)
        features = torch.zeros(self.batch_size, self.num_keypoints, 32, 32)
        heatmaps = transporter._gaussian_map(features, self.config['model']['gaussian_map_std'])
        self.assertEqual(heatmaps.shape, (self.batch_size, self.num_keypoints, 32, 32))

    def test_transport(self):
        N, K, H, W, D = 1, 2, 32, 32, 4
        transporter = Transporter(self.config)
        source_keypoints = torch.zeros(N, K, H, W)
        target_keypoints = torch.zeros(N, K, H, W)
        source_features = torch.zeros(N, D, H, W)
        target_features = torch.zeros(N, D, H, W)
        transported = transporter._transport(
            source_keypoints,
            target_keypoints,
            source_features,
            target_features
        )
        self.assertEqual(transported.shape, target_features.shape)


if __name__ == "__main__":
    unittest.main()
