import torch
import torch.nn.functional as F


def perception_loss(perception_net: torch.nn.Module,
                    prediction: torch.Tensor,
                    target: torch.Tensor) -> torch.Tensor:
    """ Uses the encodings of a perception network
        (E.g. InceptionNet net or AlexNet)
        to calculate the loss between
        prediction and target.
    """
    assert target.shape == prediction.shape
    N, T = target.shape[0], target.shape[1]
    # alpha = 0.1
    pred_encoding = perception_net(prediction.view((N*T, *prediction.shape[2:])))
    tar_encoding = perception_net(target.view((N*T, *target.shape[2:])))
    return F.mse_loss(input=pred_encoding, target=tar_encoding, reduction='sum') * 0.5
    # return torch.mean(alpha * torch.sqrt(torch.norm(pred_encoding - tar_encoding, 2)))
