"""

    This script implements the functionalities of
    https://github.com/google-research/google-research/blob/master/video_structure/ops.py
    in PyTorch

"""
import torch
import torch.nn as nn

EPSILON = 1e-6


def make_pixel_grid(axis: int, width: int):
    """ Creates linspace of length 'width' for a given axis. """
    if axis == 2:
        return torch.linspace(start=-1.0, end=1.0, steps=width)
    elif axis == 3:
        return torch.linspace(start=1.0, end=-1.0, steps=width)
    else:
        raise ValueError(f"Can not make pixel grid for axis {axis}!")


def add_coord_channels(image_tensor: torch.Tensor) -> torch.Tensor:
    """ Adds channels containing pixel indices (x and y coordinates) to an image. """
    B, C, H, W = image_tensor.shape[0], image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3]
    x_grid = torch.linspace(start=-1.0, end=1.0, steps=H).view((1, 1, H, 1))
    x_map = torch.tile(x_grid, (B, 1, 1, W))
    y_grid = torch.linspace(start=1.0, end=-1.0, steps=W).view((1, 1, 1, W))
    y_map = torch.tile(y_grid, (B, 1, H, 1))
    return torch.cat([image_tensor, x_map, y_map], dim=1)


class FeatureMapsToKeyPoints(nn.Module):

    """
        (N, C, H, W) feature maps -> (N, 2 (x, y), 1 (mu)) key-points.
    """

    def __init__(self):
        super(FeatureMapsToKeyPoints, self).__init__()
        self.map_to_x = FeatureMapsToCoordinates(axis=2)
        self.map_to_y = FeatureMapsToCoordinates(axis=3)

    def forward(self, feature_maps: torch.Tensor) -> torch.Tensor:

        # Check for non-negativity
        assert torch.min(feature_maps) >= 0

        x_coordinates = self.map_to_x(feature_maps)
        y_coordinates = self.map_to_y(feature_maps)
        map_scales = torch.mean(feature_maps, dim=[2, 3])

        # Normalize map scales to [0.0, 1.0] across key-points
        map_scales /= (EPSILON + torch.max(map_scales, dim=1, keepdim=True)[0])

        return torch.stack([x_coordinates, y_coordinates, map_scales], dim=1)


class FeatureMapsToCoordinates(nn.Module):

    """
        Reduces heatmaps to coordinates along one axis
    """
    def __init__(self, axis: int):
        self.axis = axis
        super(FeatureMapsToCoordinates, self).__init__()

    def forward(self, maps: torch.Tensor) -> torch.Tensor:

        width = maps.shape[self.axis]
        grid = make_pixel_grid(axis=self.axis, width=width)
        shape = [1, 1, 1, 1]
        shape[self.axis] = -1
        grid = grid.view(tuple(shape))

        if self.axis == 2:
            marginalize_dim = 3
        elif self.axis == 3:
            marginalize_dim = 2
        else:
            raise ValueError(f"Can not make coordinates for axis {self.axis}!")

        # Normalize heatmaps to a prob. distr. (Sum to 1)
        weights = torch.sum(maps + EPSILON, dim=marginalize_dim, keepdim=True)
        weights /= torch.sum(weights, dim=self.axis, keepdim=True)

        # Computer center of mass of marginalized maps to obtain scalar coordinates:
        coordinates = torch.sum(weights*grid, dim=self.axis, keepdim=True)

        # return coordinates.squeeze(dim=[2, 3])
        return coordinates.squeeze(-1).squeeze(-1)


class KeyPointsToFeatureMaps(nn.Module):
    """
        Creates feature maps from key-points
        with a 'gaussian blob' around their
        (x, y) - coordinates.

    """
    def __init__(self, sigma: float = 1.0, heatmap_width: int = 16):
        """ Creates class instance.

        :param sigma: Standard deviation of the 'gaussian blob', in heatmap pixels
        :param heatmap_width: Pixel width of created heatmaps
        """
        super(KeyPointsToFeatureMaps, self).__init__()
        self.sigma = sigma
        self.heatmap_width = heatmap_width

    def get_grid(self, axis: int):
        grid = make_pixel_grid(axis, self.heatmap_width)
        shape = [1, 1, 1, 1]
        shape[axis] = -1
        return grid.view(tuple(shape))

    def forward(self, keypoint_tensor: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the module.

        :param keypoint_tensor: Tensor in (N, C, 3)
        :return:
        """
        # Split scales (mu) and 2D key-point coordinates
        N, C = keypoint_tensor.shape[0], keypoint_tensor.shape[1]
        keypoint_coordinates, scales = torch.split(keypoint_tensor, [2, 1], dim=2)
        assert tuple(keypoint_coordinates.shape) == (N, C, 2)
        assert tuple(scales.shape) == (N, C, 1)

        # Expand to (B, 1, 1, C, 1)
        x_coordinates = keypoint_coordinates[:, :, 0].view(N, 1, 1, C, 1)
        y_coordinates = keypoint_coordinates[:, :, 1].view(N, 1, 1, C, 1)

        # Make 'gaussian blobs'
        keypoint_width = 2.0 * (self.sigma / self.heatmap_width) ** 2
        x_vec = torch.exp(-torch.square(self.get_grid(axis=2) - x_coordinates)/keypoint_width)
        y_vec = torch.exp(-torch.square(self.get_grid(axis=3) - y_coordinates)/keypoint_width)
        maps = torch.multiply(x_vec, y_vec)

        return maps * scales.view(N, 1, 1, C, 1)




