import os
from os.path import join
import yaml
import argparse

import torch

from src.data.mime.mime_hd_kinect_dataset import MimeHDKinectRGB
from src.agents.tcn_agent import TCN_Agent

parser = argparse.ArgumentParser()
parser.add_argument('config_path', metavar='CP', type=str, help='Path to config file.')
parser.add_argument('data_path', metavar='DP', type=str, help='Base path to dataset.')
args = parser.parse_args()

print(f"Found {torch.cuda.device_count()} available cuda devices.")

with open(args.config_path, 'r') as stream:
	tcn_conf = yaml.safe_load(stream)

data_set = MimeHDKinectRGB(
    base_path=join(args.data_path),
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
