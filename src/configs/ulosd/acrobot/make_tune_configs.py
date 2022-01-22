""" This script creates individual, counted config files under ./tuned/
    for each parameter combination in the loops.
"""

import yaml
import os

os.makedirs("./tune/", exist_ok=False)

counter = 0

with open('ulosd_acrobot_vanilla.yml', 'r') as stream:
    ulosd_conf = yaml.safe_load(stream)

for n_frames in [4, 8]:
    for n_init_filters in [4, 8, 16, 32]:
        for sig in [1.5, 2.5, 3.5]:
            for n_convs in [1, 2]:
                for init_lr in [0.1, 0.01, 0.001]:
                    ulosd_conf['model']['n_frames'] = n_frames
                    ulosd_conf['model']['n_init_filters'] = n_init_filters
                    ulosd_conf['model']['feature_map_gauss_sigma'] = sig
                    ulosd_conf['model']['n_convolutions_per_res'] = n_convs
                    ulosd_conf['training']['initial_lr'] = init_lr
                    with open(f'tune/{counter}.yml', 'w') as stream:
                        yaml.dump(ulosd_conf, stream)
                    counter += 1
