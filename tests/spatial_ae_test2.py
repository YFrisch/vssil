""" This script is used to test the agent class for the Deep Spatial Auto-Encoder."""
import os
from os.path import join

import torch.utils.data

import yaml

from src.agents.deep_spatial_ae_agent import SpatialAEAgent
from old.src.data.mime import MimeHDKinectRGB
from src.data.utils import play_video
from src.utils.visualization import make_annotated_tensor

cwd = os.getcwd()

data_set = MimeHDKinectRGB(
    base_path=join(cwd, '/datasets/'),
    tasks='stir',
    start_ind=0,
    stop_ind=10,
    img_scale_factor=0.25,
    timesteps_per_sample=10,  # -1 to sample full trajectories
    overlap=0

)

eval_data_set = MimeHDKinectRGB(
    base_path=join(cwd, '/datasets/'),
    tasks='stir',
    start_ind=0,
    stop_ind=2,
    img_scale_factor=0.25,
    timesteps_per_sample=-1,  # Sample full trajectories
    overlap=0
)

dsae_conf = yaml.safe_load(open(join(cwd, 'src/configs/deep_spatial_ae.yml')))
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
