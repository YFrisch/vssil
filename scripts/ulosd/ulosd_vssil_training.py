import os
import yaml

from torchvision import transforms

from src.utils.argparse import parse_arguments
from src.agents.ulosd_agent import ULOSD_Agent
from src.data.video_dataset import VideoFrameDataset, ImglistToTensor

if __name__ == "__main__":

    args = parse_arguments()

    with open(args.config, 'r') as stream:
        ulosd_conf = yaml.safe_load(stream)
        if ulosd_conf['warm_start']:
            with open(ulosd_conf['warm_start_config'], 'r') as stream2:
                old_conf = yaml.safe_load(stream2)
                ulosd_conf['log_dir'] = old_conf['log_dir'][:-1] + "_resume/"
        else:
            ulosd_conf['log_dir'] = ulosd_conf['log_dir']+f"/{args.id}/"
        print(ulosd_conf['log_dir'])
        ulosd_conf['data']['path'] = args.data

    # Apply any number of torchvision transforms here as pre-processing
    preprocess = transforms.Compose([
        # NOTE: The first transform already converts the image range to (0, 1)
        ImglistToTensor(),
        transforms.RandomChoice([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
        #    transforms.RandomApply([transforms.RandomRotation(degrees=90)], p=0.3),
            # transforms.RandomApply([transforms.ColorJitter(brightness=.5, hue=.3)], p=0.3)
        ])
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    prep = ImglistToTensor()

    data_set = VideoFrameDataset(
        root_path=args.data,
        annotationfile_path=os.path.join(args.data, 'annotations.txt'),
        num_segments=1,
        frames_per_segment=ulosd_conf['model']['n_frames'],
        imagefile_template='img_{:05d}.jpg',
        transform=prep,
        random_shift=True,
        test_mode=False
    )

    ulosd_agent = ULOSD_Agent(dataset=data_set,
                              config=ulosd_conf)

    ulosd_agent.train(config=ulosd_conf)
