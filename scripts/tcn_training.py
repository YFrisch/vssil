import yaml

from old.src.data.mime.mime_hd_kinect_dataset import MimeHDKinectRGB
from src.agents.tcn_agent import TCN_Agent
from src.utils.argparse import parse_arguments

args = parse_arguments()

with open(args.config_path, 'r') as stream:
    tcn_conf = yaml.safe_load(stream)

data_set = MimeHDKinectRGB(
    base_path=args.data_path,
    tasks=tcn_conf['data']['tasks'],
    timesteps_per_sample=tcn_conf['model']['n_frames'],  # 10
    overlap=tcn_conf['data']['overlap'],
    img_scale_factor=tcn_conf['data']['img_scale_factor']
)

tcn_agent = TCN_Agent(
    dataset=data_set,
    config=tcn_conf
)

tcn_agent.train(config=tcn_conf)
