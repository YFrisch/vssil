""" This script is used to test the agent class for the Deep Spatial Auto-Encoder."""


import yaml

from src.agents.deep_spatial_ae_agent import SpatialAEAgent
from src.data.mime import MimeHDKinectRGB

data_set = MimeHDKinectRGB(
    base_path='/home/yannik/vssil/datasets/',
    tasks='stir',
    start_ind=0,
    stop_ind=-1,
    img_scale_factor=0.25,
    timesteps_per_sample=10,  # -1 to sample full trajectories
    overlap=0
)

dsae_conf = yaml.safe_load(open('src/configs/deep_spatial_ae.yml'))
dsae_agent = SpatialAEAgent(dataset=data_set,
                            config=dsae_conf)

dsae_agent.train(config=dsae_conf)
# dsae_agent.evaluate(dataset=eval_data_set, config=dsae_conf)
