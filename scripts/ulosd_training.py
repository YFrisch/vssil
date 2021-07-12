import yaml

from src.utils.argparse import parse_arguments
from src.data.utils import combine_mime_hd_kinect_tasks
from src.agents.ulosd_agent import ULOSD_Agent

args = parse_arguments()

with open(args.config, 'r') as stream:
    ulosd_conf = yaml.safe_load(stream)
    ulosd_conf['log_dir'] = ulosd_conf['log_dir']+f"/{args.id}/"
    print(ulosd_conf['log_dir'])

data_set = combine_mime_hd_kinect_tasks(
    task_list=ulosd_conf['data']['tasks'],
    base_path=args.data,
    timesteps_per_sample=ulosd_conf['model']['n_frames'],
    overlap=ulosd_conf['data']['overlap'],
    img_shape=eval(ulosd_conf['data']['img_shape'])
)

ulosd_agent = ULOSD_Agent(dataset=data_set,
                          config=ulosd_conf)

ulosd_agent.train(config=ulosd_conf)
