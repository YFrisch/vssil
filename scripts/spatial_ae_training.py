""" This script is used to test the agent class for the Deep Spatial Auto-Encoder."""
import yaml

from src.agents.deep_spatial_ae_agent import SpatialAEAgent
from old.src.data.mime import MimeHDKinectRGB
from src.utils.argparse import parse_arguments

args = parse_arguments()

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
