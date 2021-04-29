import torch


def make_annotated_tensor(image_series: torch.Tensor, feature_positions: torch.Tensor):
    """ Modifies the input tensor cointaining a sequence of images to
        be annotated at the given feature positions
    :param image_series: Torch tensor of image series in (T, C, H, W) format
    :param feature_positions: Torch tensor of 2D feature positions in (T, F, 2) format
    """

    new_img_series = torch.clone(image_series)
    size = 2
    for t in range(image_series.shape[0]):
        for f in range(feature_positions.shape[1]):
            feature_pos_2d = feature_positions[t, f, :].int()
            x_start = feature_pos_2d[0] - size
            x_stop = feature_pos_2d[0] + size
            y_start = feature_pos_2d[1] - size
            y_stop = feature_pos_2d[1] + size
            new_img_series[t, 0, x_start:x_stop, y_start:y_stop] = 0  # R
            new_img_series[t, 1, x_start:x_stop, y_start:y_stop] = 1  # G
            new_img_series[t, 2, x_start:x_stop, y_start:y_stop] = 0  # B

    return new_img_series

