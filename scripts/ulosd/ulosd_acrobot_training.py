import yaml

import torch
import torchvision.transforms as transforms

from src.utils.argparse import parse_arguments
from src.agents.ulosd_agent import ULOSD_Agent
from src.data.npz_dataset import NPZ_Dataset

if __name__ == "__main__":

    # NOTE: This line might produce non-deterministic results
    torch.backends.cudnn.benchmark = True

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
        ulosd_conf['data']['path'] = args.data

    tran = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomHorizontalFlip(p=0.9),
            transforms.RandomVerticalFlip(p=0.9),
            #transforms.RandomApply([transforms.RandomRotation(degrees=90)], p=0.3),
        ]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    npz_data_set = NPZ_Dataset(
        num_timesteps=ulosd_conf['model']['n_frames'],
        root_path=args.data,
        key_word='images',
        transform=None
    )

    ulosd_agent = ULOSD_Agent(dataset=npz_data_set,
                              config=ulosd_conf)

    ulosd_agent.train(config=ulosd_conf)
