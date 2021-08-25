import yaml

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.utils.argparse import parse_arguments
from src.models.transporter_keypointer import TransporterKeypointer
from src.models.utils import init_weights

args = parse_arguments()

with open(args.config, 'r') as stream:
    transporter_conf = yaml.safe_load(stream)
    if transporter_conf['warm_start']:
        with open(transporter_conf['warm_start_config'], 'r') as stream2:
            old_conf = yaml.safe_load(stream2)
            transporter_conf['log_dir'] = old_conf['log_dir'][:-1] + "_resume/"
    else:
        transporter_conf['log_dir'] = transporter_conf['log_dir']+f"/{args.id}/"
    transporter_conf['device'] = 'cpu'

key_pointer = TransporterKeypointer(config=transporter_conf)
key_pointer.apply(lambda model: init_weights(m=model, config=transporter_conf))

fake_img = torch.ones(size=(1, 3, 64, 64))
fake_img_feature_maps = key_pointer.net(fake_img)
fake_keypoint_feature_maps = key_pointer.regressor(fake_img_feature_maps)
fake_keypoint_means, fake_gaussian_maps = \
    key_pointer.feature_maps_to_keypoints(feature_map=fake_keypoint_feature_maps)
print(fake_gaussian_maps.shape)
print(fake_keypoint_means)

plt.figure()
for m in range(fake_gaussian_maps.shape[1]):
    plt.imshow(fake_gaussian_maps[0, m, ...].detach().cpu().numpy())
    plt.show()
