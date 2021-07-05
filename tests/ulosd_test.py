import os
from os.path import join
import yaml

from src.data.mime.mime_hd_kinect_dataset import MimeHDKinectRGB
from src.agents.ulosd_agent import ULOSD_Agent

cwd = os.getcwd()

data_set = MimeHDKinectRGB(
    base_path=join(cwd, 'data/datasets/'),
    timesteps_per_sample=5,
    overlap=0,
    img_scale_factor=0.25
)

sample = data_set.__getitem__(0)
sample = sample.unsqueeze(0)
print(sample.shape)

ulosd_conf = yaml.safe_load(open(join(cwd, 'src/configs/ulosd.yml')))
ulosd_agent = ULOSD_Agent(dataset=data_set,
                          config=ulosd_conf)

feature_points = ulosd_agent.model(sample)



#ulosd_agent.train(config=ulosd_conf)
