import yaml
import argparse
from datetime import datetime

from src.data.utils import combine_mime_hd_kinect_tasks
from src.agents.ulosd_agent import ULOSD_Agent

parser = argparse.ArgumentParser()
parser.add_argument('config_path', metavar='CP', type=str, help='Path to config file.')
parser.add_argument('data_path', metavar='DP', type=str, help='Base path to dataset.')
year, month, day, hour, minute = datetime.now().year, datetime.now().month, datetime.now().day, \
                                         datetime.now().hour, datetime.now().minute
parser.add_argument('id', metavar='ID', type=str, help='Individual id of the experiment.',
                    default=f"/{year}_{month}_{day}_{hour}_{minute}/")
args = parser.parse_args()

with open(args.config_path, 'r') as stream:
    ulosd_conf = yaml.safe_load(stream)
    ulosd_conf['log_dir'] = ulosd_conf['log_dir']+f"/{args.id}"

data_set = combine_mime_hd_kinect_tasks(
    task_list=ulosd_conf['data']['tasks'],
    base_path=args.data_path,
    timesteps_per_sample=ulosd_conf['model']['n_frames'],
    overlap=ulosd_conf['data']['overlap'],
    img_shape=eval(ulosd_conf['data']['img_shape'])
)

ulosd_agent = ULOSD_Agent(dataset=data_set,
                          config=ulosd_conf)

ulosd_agent.train(config=ulosd_conf)
