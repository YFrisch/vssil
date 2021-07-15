import torch

from src.models.inception3 import CustomInception3


def inception_encoding_loss(inception_net: CustomInception3,
                            prediction: torch.Tensor,
                            target: torch.Tensor) -> torch.Tensor:
    """ Uses the encodings of InceptionNet to calculate the loss between
        prediction and target.
    """
    assert target.shape == prediction.shape
    N, T = target.shape[0], target.shape[1]
    alpha = 0.1
    pred_encoding = inception_net(prediction.view((N*T, *prediction.shape[2:])))
    tar_encoding = inception_net(target.view((N*T, *target.shape[2:])))
    return torch.mean(alpha * torch.sqrt(torch.norm(pred_encoding - tar_encoding, 2)))
