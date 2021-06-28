""" This script is used to test the agent class for the Deep Spatial Auto-Encoder."""
import os
from os.path import join
import yaml
import argparse

from src.agents.deep_spatial_ae_agent import SpatialAEAgent
from src.data.mime import MimeHDKinectRGB

parser = argparse.ArgumentParser()
parser.add_argument('config_path', metavar='CP', type=str, help='Path to config file.')
parser.add_argument('data_path', metavar='DP', type=str, help='Base path to dataset.')
args = parser.parse_args()

with open(args.config_path, 'r') as stream:
	dsae_conf = yaml.safe_load(stream)

data_set = MimeHDKinectRGB(
    base_path=args.data_path,
    tasks='stir',
    start_ind=0,
    stop_ind=-1,
    img_scale_factor=0.5,
    timesteps_per_sample=10,  # -1 to sample full trajectories
    overlap=0
)

dsae_agent = SpatialAEAgent(dataset=data_set,
                            config=dsae_conf)
dsae_agent.train(config=dsae_conf)
