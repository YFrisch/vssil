import torch
import yaml

from torch.utils.data import DataLoader

from src.agents.transporter_agent import TransporterAgent
from src.data.npz_dataset import NPZ_Dataset
from src.utils.argparse import parse_arguments
from src.utils.visualization import play_series_and_reconstruction_with_keypoints

if __name__ == "__main__":

    args = parse_arguments()
    args.config = '/home/yannik/vssil/results/transporter_acrobot/2021_12_5_22_3/config.yml'

    with open(args.config, 'r') as stream:
        transporter_conf = yaml.safe_load(stream)
        if transporter_conf['warm_start']:
            with open(transporter_conf['warm_start_config'], 'r') as stream2:
                old_conf = yaml.safe_load(stream2)
                transporter_conf['log_dir'] = old_conf['log_dir'][:-1] + "_resume/"
        else:
            transporter_conf['log_dir'] = transporter_conf['log_dir']+f"/{args.id}/"
        print(transporter_conf['log_dir'])
        transporter_conf['multi_gpu'] = False
        transporter_conf['device'] = 'cpu'

    npz_data_set = NPZ_Dataset(
        num_timesteps=200,  # 200
        root_path='/home/yannik/vssil/video_structure/testdata/acrobot_swingup_random_repeat40_00006887be28ecb8.npz',
        key_word='images'
    )

    eval_data_loader = DataLoader(
        dataset=npz_data_set,
        batch_size=1,
        shuffle=True
    )

    transporter_agent = TransporterAgent(dataset=npz_data_set, config=transporter_conf)
    transporter_agent.load_checkpoint(
        '/home/yannik/vssil/results/transporter_acrobot/2021_12_5_22_3/checkpoints/chckpt_f0_e195.PTH')

    print("##### Evaluating:")
    with torch.no_grad():
        for i, (sample, label) in enumerate(eval_data_loader):

            samples = None
            reconstructed_diffs = None
            reconstructions = None
            key_points = None

            t_diff = 1

            for t in range(sample.shape[1] - t_diff):
                print(f'{t}|{sample.shape[1] - t_diff}')
                _sample, target = transporter_agent.preprocess(sample[:, t:t+1+t_diff, ...], label, transporter_conf)
                _sample.to(transporter_agent.device)
                target.to(transporter_agent.device)

                reconstruction = transporter_agent.model(_sample, target).clip(-0.5, 0.5)
                reconstructed_diff = (reconstruction - _sample).clip(-1.0, 1.0)
                target_diff = (target - _sample).clip(-1.0, 1.0)
                key_point_coordinates = transporter_agent.model.keypointer(_sample)[0]
                # Adapt to visualization
                key_point_coordinates[..., 1] *= -1

                samples = _sample.unsqueeze(1) if samples is None else torch.cat([samples, _sample.unsqueeze(1)], dim=1)
                reconstructions = reconstruction.unsqueeze(1) if reconstructions is None\
                    else torch.cat([reconstructions, reconstruction.unsqueeze(1)], dim=1)
                key_points = key_point_coordinates.unsqueeze(1) if key_points is None \
                    else torch.cat([key_points, key_point_coordinates.unsqueeze(1)], dim=1)

            play_series_and_reconstruction_with_keypoints(image_series=samples,
                                                          reconstruction=reconstructions,
                                                          keypoint_coords=key_points)
            exit()
