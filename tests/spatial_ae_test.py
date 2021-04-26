import os

import yaml
from torch.utils.data import DataLoader

from agents.spatial_ae_agent import SpatialAEAgent
from models.deep_spatial_autoencoder import DeepSpatialAE
from data.mime_dataset_wrapper import MimeDataSet
from data.utils import play_video

device = "cuda:0"

data_set = MimeDataSet(
    base_path='/home/yannik/vssil/data/datasets/',
    task='stir',
    joint_data=False,
    hd_kinect_img_data=True,
    img_scale_factor=0.2
)

data_loader = DataLoader(
    data_set,
    batch_size=1,
    shuffle=True
)

print(os.getcwd())
dsae_conf = yaml.safe_load(open('configs/deep_spatial_ae.yml'))
dsae_agent = SpatialAEAgent(config=dsae_conf)
dsae_agent.train_data_loader = data_loader
dsae_agent.eval_data_loader = data_loader
dsae_agent.train(config=dsae_conf)
# dsae_agent.evaluate(config=dsae_conf)
