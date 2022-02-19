""" This script is used to test the agent class for the Deep Spatial Auto-Encoder."""
import yaml

from src.agents.dsae_agent import DSAEAgent
from src.utils.argparse import parse_arguments
from src.data.utils import get_dataset_from_path

args = parse_arguments()

with open(args.config, 'r') as stream:
    dsae_conf = yaml.safe_load(stream)
    if dsae_conf['warm_start']:
        with open(dsae_conf['warm_start_config'], 'r') as stream2:
            old_conf = yaml.safe_load(stream2)
            dsae_conf['log_dir'] = old_conf['log_dir'][:-1] + "_resume/"
    else:
        dsae_conf['log_dir'] = dsae_conf['log_dir'] + f"/{args.id}/"
    print(dsae_conf['log_dir'])
    dsae_conf['data']['path'] = args.data

data_set = get_dataset_from_path(
    root_path=args.data,
    n_timesteps=dsae_conf['model']['n_frames']
)

dsae_agent = DSAEAgent(dataset=data_set, config=dsae_conf)

dsae_agent.train(config=dsae_conf)
