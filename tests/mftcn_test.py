import os
from os.path import join
import yaml

from old.src.data.mime.mime_hd_kinect_dataset import MimeHDKinectRGB
from src.agents.mf_tcn_agent import MF_TCN_Agent

cwd = os.getcwd()

data_set = MimeHDKinectRGB(
    base_path=join(cwd, '/datasets/'),
    timesteps_per_sample=5,
    overlap=0,
    img_scale_factor=0.25
)

sample = data_set.__getitem__(0)
print(sample.shape)

mf_tcn_conf = yaml.safe_load(open(join(cwd, 'src/configs/mf_tcn.yml')))
mf_tcn_agent = MF_TCN_Agent(dataset=data_set,
                            config=mf_tcn_conf)

mf_tcn_agent.train(config=mf_tcn_conf)
