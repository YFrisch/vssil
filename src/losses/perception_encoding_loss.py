import torch
import torch.nn.functional as F


def perception_loss(perception_net: torch.nn.Module,
                    prediction: torch.Tensor,
                    target: torch.Tensor) -> torch.Tensor:
    """ Uses the encodings of a pre-trained perception network (E.g. InceptionNet net or AlexNet)
        to calculate the loss between prediction and target.

    :param perception_net: Perceptual network to use to encode prediction & target.
    :param prediction: Torch tensor of sequential image frame predictions in (N, T, C, H, W)
    :param target: Torch tensor of sequential images in (N, T, C, H, W)
    """
    assert target.shape == prediction.shape
    N, T = target.shape[0], target.shape[1]
    prediction_encoding = perception_net(prediction.view((N*T, *prediction.shape[2:])))
    target_encoding = perception_net(target.view((N*T, *target.shape[2:])))
    return F.mse_loss(input=prediction_encoding, target=target_encoding, reduction='sum') / (2*N)
    # return F.mse_loss(input=prediction_encoding, target=target_encoding, reduction='mean')
