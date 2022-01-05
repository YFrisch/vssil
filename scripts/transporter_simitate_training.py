import os
import yaml

from torchvision import transforms

from src.agents.transporter_agent import TransporterAgent
from src.data.video_dataset import VideoFrameDataset, ImglistToTensor
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

    preprocess = transforms.Compose([
            ImglistToTensor(),
        ])

    data_set = VideoFrameDataset(
            root_path=args.data,
            annotationfile_path=os.path.join(args.data, 'annotations.txt'),
            num_segments=1,
            frames_per_segment=transporter_conf['model']['n_frames'],
            imagefile_template='img_{:05d}.jpg',
            transform=preprocess,
            random_shift=True,
            test_mode=False
        )

    transporter_agent = TransporterAgent(dataset=data_set, config=transporter_conf)

    transporter_agent.train(config=transporter_conf)
