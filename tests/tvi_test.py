import yaml
import os
from os.path import join

from src.data.mime import MimeJointAngles
from src.agents.tvi_agent import TVI_Agent

cwd = os.getcwd()

tvi_conf = yaml.safe_load(open(join(cwd, 'src/configs/tvi.yml')))

dataset = MimeJointAngles(
    base_path=join(cwd, "datasets"),
    tasks="stir",
    start_ind=0,
    stop_ind=-1,
    timesteps_per_sample=1,
    overlap=0
)

lstm = TVI_Agent(
    dataset=dataset,
    config=tvi_conf
)
