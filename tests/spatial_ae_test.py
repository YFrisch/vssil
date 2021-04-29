""" This script is used to test the agent class for the Deep Spatial Auto-Encoder."""


import yaml

from src.agents.deep_spatial_ae_agent import SpatialAEAgent
from src.data.mime.mime_dataset_wrapper import MimeDataSet

data_set = MimeDataSet(
    base_path='/home/yannik/vssil/datasets/',
    task='stir',
    start_ind=0,
    stop_ind=-1,
    joint_data=False,
    hd_kinect_img_data=True,
    rd_kinect_img_data=False,
    img_scale_factor=0.25,
    timesteps_per_sample=5,  # -1 to sample full trajectories
)

eval_data_set = MimeDataSet(
    base_path='/home/yannik/vssil/datasets/',
    task='stir',
    start_ind=0,
    stop_ind=-1,
    joint_data=False,
    hd_kinect_img_data=True,
    rd_kinect_img_data=False,
    img_scale_factor=0.25,
    timesteps_per_sample=-1,  # Sample full trajectories
)

dsae_conf = yaml.safe_load(open('src/configs/deep_spatial_ae.yml'))
dsae_agent = SpatialAEAgent(dataset=data_set,
                            config=dsae_conf)

dsae_agent.train(config=dsae_conf)
# dsae_agent.evaluate(dataset=eval_data_set, config=dsae_conf)
