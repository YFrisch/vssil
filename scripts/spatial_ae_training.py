""" This script is used to test the agent class for the Deep Spatial Auto-Encoder."""
import os
from os.path import join
import yaml

from src.agents.deep_spatial_ae_agent import SpatialAEAgent
from src.data.mime import MimeHDKinectRGB

cwd = os.getcwd()

dsae_conf = yaml.safe_load(open(join(cwd, 'src/configs/deep_spatial_ae.yml')))

data_set = MimeHDKinectRGB(
    base_path=join(cwd, 'datasets'),
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
