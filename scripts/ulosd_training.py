import os
from os.path import join
import yaml

import matplotlib.pyplot as plt

from src.data.utils import combine_mime_hd_kinect_tasks
from src.agents.ulosd_agent import ULOSD_Agent

cwd = os.getcwd()

ulosd_conf = yaml.safe_load(open(join(cwd, 'src/configs/ulosd.yml')))

data_set = combine_mime_hd_kinect_tasks(
    task_list=ulosd_conf['data']['tasks'],
    base_path=join(cwd, 'datasets/'),
    timesteps_per_sample=ulosd_conf['model']['n_frames'],
    overlap=ulosd_conf['data']['overlap'],
    img_scale_factor=eval(ulosd_conf['data']['img_scale_factors'])
)

ulosd_agent = ULOSD_Agent(dataset=data_set,
                          config=ulosd_conf)

ulosd_agent.train(config=ulosd_conf)
