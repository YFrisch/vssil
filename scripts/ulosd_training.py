import os
from os.path import join
import yaml

import matplotlib.pyplot as plt

from src.data.mime.mime_hd_kinect_dataset import MimeHDKinectRGB
from src.agents.ulosd_agent import ULOSD_Agent

cwd = os.getcwd()

ulosd_conf = yaml.safe_load(open(join(cwd, 'src/configs/ulosd.yml')))


data_set = MimeHDKinectRGB(
    base_path=join(cwd, 'datasets/'),
    timesteps_per_sample=ulosd_conf['model']['n_frames'],
    overlap=0,
    img_scale_factor=(160/240, 160/640)
)

ulosd_agent = ULOSD_Agent(dataset=data_set,
                          config=ulosd_conf)

ulosd_agent.train(config=ulosd_conf)
