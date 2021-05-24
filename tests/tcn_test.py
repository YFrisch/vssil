import torch
import yaml

from torch.utils.data import DataLoader

from src.data.mime.mime_hd_kinect_dataset import MimeHDKinectRGB
from src.agents.tcn_agent import TCN_Agent

tcn_conf = yaml.safe_load(open('src/configs/tcn.yml'))

data_set = MimeHDKinectRGB(
    base_path='/home/yannik/vssil/datasets',
    timesteps_per_sample=tcn_conf['model']['n_frames'],  # 10
    overlap=0,
    img_scale_factor=0.25
)

tcn_agent = TCN_Agent(dataset=data_set,
                      config=tcn_conf)

tcn_agent.train(config=tcn_conf)
