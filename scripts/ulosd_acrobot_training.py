import yaml

import torch
import torchvision.transforms as transforms

from src.utils.argparse import parse_arguments
from src.agents.ulosd_agent import ULOSD_Agent
from src.data.npz_dataset import NPZ_Dataset

if __name__ == "__main__":

    # NOTE: This line might produce non-deterministic results
    torch.backends.cudnn.benchmark = True
    #torch.autograd.set_detect_anomaly(True)

    args = parse_arguments()

    with open(args.config, 'r') as stream:
        ulosd_conf = yaml.safe_load(stream)
        if ulosd_conf['warm_start']:
            with open(ulosd_conf['warm_start_config'], 'r') as stream2:
                old_conf = yaml.safe_load(stream2)
                ulosd_conf['log_dir'] = old_conf['log_dir'][:-1] + "_resume/"
        else:
            ulosd_conf['log_dir'] = ulosd_conf['log_dir'] + f"/{args.id}/"
        print(ulosd_conf['log_dir'])
        ulosd_conf['multi_gpu'] = False

    tran = transforms.RandomApply([
        transforms.ColorJitter(brightness=.5, hue=.3),
        transforms.RandomAdjustSharpness(sharpness_factor=2.0),
        # transforms.RandomEqualize(),
        transforms.RandomHorizontalFlip(p=0.9),
        transforms.RandomVerticalFlip(p=0.9)
    ])

    npz_data_set = NPZ_Dataset(
        num_timesteps=ulosd_conf['model']['n_frames'],
        root_path='/home/yannik/vssil/video_structure/testdata/acrobot_swingup_random_repeat40_00006887be28ecb8.npz',
        key_word='images',
        transform=tran
    )

    ulosd_agent = ULOSD_Agent(dataset=npz_data_set,
                              config=ulosd_conf)

    ulosd_agent.train(config=ulosd_conf)
