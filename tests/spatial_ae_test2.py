""" This script is used to test the agent class for the Deep Spatial Auto-Encoder."""
import os

import torch.utils.data

print(os.getcwd())

import yaml

from src.agents.deep_spatial_ae_agent import SpatialAEAgent
from src.data.mime.mime_dataset_wrapper import MimeDataSet
from src.data.utils import play_video
from src.utils.feature_visualization import make_annotated_tensor

data_set = MimeDataSet(
    base_path='/home/yannik/vssil/datasets/',
    task='stir',
    start_ind=0,
    stop_ind=10,
    joint_data=False,
    hd_kinect_img_data=True,
    rd_kinect_img_data=False,
    img_scale_factor=0.25,
    timesteps_per_sample=10,  # -1 to sample full trajectories
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

# dsae_agent.load_checkpoint(dsae_conf['evaluation']['chckpt_path'])

data_loader = torch.utils.data.DataLoader(eval_data_set, batch_size=1)

with torch.no_grad():

    for i, sample in enumerate(data_loader):

        sample, target = dsae_agent.preprocess(x=sample, config=dsae_conf)
        sample, target = sample.to(dsae_agent.device), target.to(dsae_agent.device)
        print("Input: ", sample.shape)
        play_video(sample.squeeze())

        features = dsae_agent.model.encode(x=sample.squeeze())
        print("Features: ", features.shape)

        annotated_sample = make_annotated_tensor(sample.squeeze(), features.squeeze())
        play_video(annotated_sample)

        reconstructed_sample = dsae_agent.model(sample.squeeze())
        print("Reconstruction: ", reconstructed_sample.shape)
        play_video(reconstructed_sample)
        exit()
