import torch


def get_img_coordinates(h: int, w: int, normalize: bool):
    """ TODO """
    x_range = torch.arange(w, dtype=torch.float32)
    y_range = torch.arange(h, dtype=torch.float32)
    if normalize:
        x_range = (x_range / (w - 1)) * 2 - 1
        y_range = (y_range / (h - 1)) * 2 - 1
    image_x = x_range.unsqueeze(0).repeat_interleave(h, 0)
    image_y = y_range.unsqueeze(0).repeat_interleave(w, 0).t()
    return image_x, image_y
