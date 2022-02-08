import yaml

import torchvision.transforms as transforms

from src.agents.transporter_agent import TransporterAgent
from src.data.npz_dataset import NPZ_Dataset
from src.utils.argparse import parse_arguments

if __name__ == "__main__":

    args = parse_arguments()

    with open(args.config, 'r') as stream:
        transporter_conf = yaml.safe_load(stream)
        if transporter_conf['warm_start']:
            with open(transporter_conf['warm_start_config'], 'r') as stream2:
                old_conf = yaml.safe_load(stream2)
                transporter_conf['log_dir'] = old_conf['log_dir'][:-1] + "_resume/"
        else:
            transporter_conf['log_dir'] = transporter_conf['log_dir']+f"/{args.id}/"
        print(transporter_conf['log_dir'])
        transporter_conf['data']['path'] = args.data

    # Tune parameters according to -hp argument
    if args.hp:
        for hp in args.hp.split(","):
            key, val = hp.split("=")
            key_split = key.split(".")
            if len(key_split) == 2:
                transporter_conf[key_split[0]][key_split[1]] = val
            else:
                transporter_conf[key] = val

    tran = transforms.Compose([
        transforms.RandomApply([
            transforms.RandomHorizontalFlip(p=0.9),
            transforms.RandomVerticalFlip(p=0.9),
            # transforms.RandomApply([transforms.RandomRotation(degrees=90)], p=0.3),
        ]),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    npz_data_set = NPZ_Dataset(
        num_timesteps=transporter_conf['model']['n_frames'],
        root_path=args.data,
        key_word='images'
    )

    transporter_agent = TransporterAgent(dataset=npz_data_set, config=transporter_conf)

    transporter_agent.train(config=transporter_conf)
