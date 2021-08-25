import yaml

from src.agents.transporter_agent import TransporterAgent
from src.data.npz_dataset import NPZ_Dataset
from src.utils.argparse import parse_arguments

if __name__ == "__main__":

    args = parse_arguments()

    with open(args.config, 'r') as stream:
        transporter_conf = yaml.safe_load(stream)
        if transporter_conf['warm_start']:
            with open(transporter_conf['warm_start_config'], 'r') as stream2:
                old_conf = yaml.safe_load(stream2)
                transporter_conf['log_dir'] = old_conf['log_dir'][:-1] + "_resume/"
        else:
            transporter_conf['log_dir'] = transporter_conf['log_dir']+f"/{args.id}/"
        print(transporter_conf['log_dir'])

    npz_data_set = NPZ_Dataset(
        num_timesteps=transporter_conf['model']['n_frames'],
        root_path='/home/yannik/vssil/video_structure/testdata/acrobot_swingup_random_repeat40_00006887be28ecb8.npz',
        key_word='images'
    )

    transporter_agent = TransporterAgent(dataset=npz_data_set, config=transporter_conf)

    transporter_agent.train(config=transporter_conf)
