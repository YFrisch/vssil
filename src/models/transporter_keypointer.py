import torch
import torch.nn as nn

from .transporter_encoder import TransporterBlock


class TransporterKeypointer(nn.Module):

    def __init__(self,
                 config: dict):
        super(TransporterKeypointer, self).__init__()

        self.net = nn.Sequential(
            TransporterBlock(in_channels=config['model']['num_img_channels'], out_channels=32,
                             kernel_size=(7, 7), stride=(1,), padding=(3,),
                             activation=config['model']['activation'],
                             skip_connections=config['model']['skip_connections']),
            TransporterBlock(in_channels=32, out_channels=32,
                             kernel_size=(3, 3), stride=(1,), padding=(1,),
                             activation=config['model']['activation'],
                             skip_connections=config['model']['skip_connections']),
            TransporterBlock(in_channels=32, out_channels=64,
                             kernel_size=(3, 3), stride=(2,), padding=(1,),
                             activation=config['model']['activation'],
                             skip_connections=config['model']['skip_connections']),
            TransporterBlock(in_channels=64, out_channels=64,
                             kernel_size=(3, 3), stride=(1,), padding=(1,),
                             activation=config['model']['activation'],
                             skip_connections=config['model']['skip_connections']),
            TransporterBlock(in_channels=64, out_channels=128,
                             kernel_size=(3, 3), stride=(2,), padding=(1,),
                             activation=config['model']['activation'],
                             skip_connections=config['model']['skip_connections']),
            TransporterBlock(in_channels=128, out_channels=128,
                             kernel_size=(3, 3), stride=(1,), padding=(1,),
                             activation=config['model']['activation'],
                             skip_connections=config['model']['skip_connections']),
        )

        self.regressor = nn.Conv2d(
            in_channels=128, out_channels=config['model']['num_keypoints'], kernel_size=(1, 1)
        )

        self.gauss_std = config['model']['gaussian_map_std']

        self.device = config['device']

    def _keypoint_means_to_gaussian_maps(self,
                                         mean: torch.Tensor,
                                         map_size: tuple,
                                         inv_std: torch.Tensor,
                                         power: int = 2):
        """ Transforms the key-point center points to gaussian masks.

        :param mean: Normalized ([-1.0, 1.0]) key-point coordinates in (N, C, 2)
        :param map_size: Tuple of the shape of a single feature map (H', W')
        :param inv_std: Inverse of the standard deviation of the gaussian blobs
        :param power: TODO
        :return: Reconstructed gaussian feature maps in (N, C, H', W')
        """

        mean_x, mean_y = mean[..., 1].unsqueeze(-1), mean[..., 0].unsqueeze(-1)

        y = torch.linspace(start=-1.0, end=1.0, steps=map_size[0],
                           dtype=torch.float32, requires_grad=False).view(1, 1, map_size[0], 1).to(self.device)

        x = torch.linspace(start=-1.0, end=1.0, steps=map_size[1],
                           dtype=torch.float32, requires_grad=False).view(1, 1, 1, map_size[1]).to(self.device)

        mean_x, mean_y = mean_x.unsqueeze(-1), mean_y.unsqueeze(-1)

        g_x = torch.pow((x - mean_x), power)
        g_y = torch.pow((y - mean_y), power)
        inv_var = torch.pow(inv_std, power).to(self.device)
        dist = torch.mul((g_y + g_x), inv_var)
        g_yx = torch.exp(-dist)

        return g_yx

    def _feature_maps_to_coordinate(self, feature_maps: torch.Tensor, axis: int) -> torch.Tensor:
        """ Returns the key-point coordinate encoding along the given axis.

        :param feature_maps: Tensor of feature maps in (N, C, H', W')
        :param axis: Axis to extract (1 or 2)
        :return:
        """
        N, K = feature_maps.shape[0], feature_maps.shape[1]
        assert axis in [2, 3], "Axis needs to be 2 or 3!"

        other_axis = 3 if axis == 2 else 2
        axis_size = feature_maps.shape[axis]

        # Normalized weight for each row/column along the axis
        g_c_prob = torch.mean(feature_maps, dim=other_axis)
        g_c_prob = torch.softmax(g_c_prob, dim=-1)

        # Linear combination of the inverval [-1, 1] using the normalized weights
        scale = torch.linspace(start=-1.0, end=1.0, steps=axis_size,
                               dtype=torch.float32, requires_grad=False).view(1, 1, axis_size).to(self.device)

        coordinate = torch.sum(g_c_prob * scale, dim=-1)

        assert tuple(coordinate.shape) == (N, K)

        return coordinate

    def _get_keypoint_means(self, feature_maps: torch.Tensor) -> torch.Tensor:
        """ Returns the center points of the key-point feature maps.

        :param feature_maps: Tensor of feature maps in (N, C, H', W')
        :return: Key-point coordinates in (N, C, 2), normalized to a range of (-1, 1)
        """
        gauss_x = self._feature_maps_to_coordinate(feature_maps, axis=3)
        gauss_y = self._feature_maps_to_coordinate(feature_maps, axis=2)
        # TODO: Check stacking dim
        gauss_mean = torch.stack([gauss_y, gauss_x], dim=2)
        return gauss_mean

    def feature_maps_to_keypoints(self,
                                  feature_map: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """ Returns key-point information from the given feature maps.

        :param feature_map: Feature maps (encoder output) in (N, C, H', W')
        :return: Tuple of ((N, C, 2) key-point coordinates,
                            (N, C, H', W') reconstructed gaussian maps)
        """
        map_size = tuple(feature_map.shape)[2:4]
        gauss_std = torch.tensor(data=[self.gauss_std], dtype=torch.float32, requires_grad=False)
        keypoint_means = self._get_keypoint_means(feature_map)
        gaussian_maps = self._keypoint_means_to_gaussian_maps(keypoint_means, map_size, 1.0 / gauss_std)
        return keypoint_means, gaussian_maps

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        img_feature_maps = self.net(x)
        key_point_feature_maps = self.regressor(img_feature_maps)
        keypoint_means, gaussian_maps = self.feature_maps_to_keypoints(feature_map=key_point_feature_maps)
        return keypoint_means, gaussian_maps
