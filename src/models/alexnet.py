import os

import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

from src.models.utils import partial_load_state_dict

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth',
}


class CustomAlexNet(nn.Module):

    """

        Modified version of AlexNet from https://pytorch.org/vision/stable/_modules/torchvision/models/alexnet.html

        The architecture is cut after the second relu activation.

        Following https://arxiv.org/abs/2001.03444

    """

    def __init__(self, num_classes: int = 1000) -> None:
        super(CustomAlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),

        )
        """
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(192, 384, kernel_size=(3,), padding=(1,)),
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 256, kernel_size=(3,), padding=(1,)),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=(3,), padding=(1,)),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        """
        # Sigmoid layer to normalize the outputs
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.classifier(x)
        # TODO: x = self.sigmoid(x) ?
        return x


def perception_alex_net(log_dir: str):

    log_dir = 'src/models/loaded/'

    if not os.path.isfile(os.path.join(log_dir, 'alex_net.pth')):
        os.makedirs(name=log_dir, exist_ok=True)
        state_dict = load_state_dict_from_url(model_urls['alexnet'])
        torch.save(state_dict, f=os.path.join(log_dir, "alex_net.pth"))
    else:
        state_dict = torch.load(os.path.join(log_dir, 'alex_net.pth'))

    alex_net = CustomAlexNet()

    partial_load_state_dict(
        model=alex_net,
        loaded_dict=state_dict
    )

    return alex_net

