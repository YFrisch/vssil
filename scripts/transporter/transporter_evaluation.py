import yaml

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from src.agents.transporter_agent import TransporterAgent
from src.data.utils import get_dataset_from_path
from src.utils.argparse import parse_arguments
from src.utils.visualization import play_series_with_keypoints, plot_keypoint_amplitudes
from src.utils.kpt_utils import get_image_patches
from src.losses.kpt_metrics import grad_tracking_metric, visual_difference_metric, distribution_metric


if __name__ == "__main__":

    args = parse_arguments()

    with open(args.config, 'r') as stream:
        transporter_conf = yaml.safe_load(stream)
        print(transporter_conf['log_dir'])
        transporter_conf['multi_gpu'] = False
        transporter_conf['device'] = 'cpu'
        transporter_conf['model']['n_frames'] = 2

    data_set = get_dataset_from_path(args.data, n_timesteps=30)  # 150

    # Use last 10 percent of data-set for evaluation (Not seen during training)
    stop_ind = len(data_set)
    start_ind = int(stop_ind * 0.9) + 1
    # random_sampler = SubsetRandomSampler(indices=range(start_ind, stop_ind))
    random_sampler = SubsetRandomSampler(indices=[stop_ind-4])  # Only single sample

    eval_data_loader = DataLoader(
        dataset=data_set,
        batch_size=1,
        shuffle=False,
        sampler=random_sampler
    )

    transporter_agent = TransporterAgent(dataset=data_set, config=transporter_conf)
    transporter_agent.eval_data_loader = eval_data_loader

    transporter_agent.load_checkpoint(
        args.checkpoint,
        map_location='cpu'
    )

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
                _sample, target = transporter_agent.preprocess(sample[:, t:t + 1 + t_diff, ...], label,
                                                               transporter_conf)
                _sample.to(transporter_agent.device)
                target.to(transporter_agent.device)

                reconstruction = transporter_agent.model(_sample, target).clip(-0.5, 0.5)
                reconstructed_diff = (reconstruction - _sample).clip(-1.0, 1.0)
                target_diff = (target - _sample).clip(-1.0, 1.0)
                key_point_coordinates = transporter_agent.model.keypointer(_sample)[0]
                # Adapt to visualization
                key_point_coordinates[..., 1] *= -1

                samples = sample[:, t:t+1, ...] if samples is None \
                    else torch.cat([samples, sample[:, t:t+1, ...]], dim=1)
                reconstructions = reconstruction.unsqueeze(1) if reconstructions is None \
                    else torch.cat([reconstructions, reconstruction.unsqueeze(1)], dim=1)
                key_points = key_point_coordinates.unsqueeze(1) if key_points is None \
                    else torch.cat([key_points, key_point_coordinates.unsqueeze(1)], dim=1)
            
            play_series_with_keypoints(image_series=samples,
                                       keypoint_coords=key_points,
                                       key_point_trajectory=True,
                                       trajectory_length=10,
                                       save_path='./result_videos_transporter/',
                                       save_frames=True)

            plot_keypoint_amplitudes(keypoint_coordinates=key_points,
                                     target_path='./result_videos_transporter/')

            patches = get_image_patches(image_sequence=samples, kpt_sequence=key_points,
                                        patch_size=(16, 16))

            M_tracking = grad_tracking_metric(patches)
            M_visual = visual_difference_metric(patches)
            M_distribution = distribution_metric(key_points, (16, 16))

            with open('./result_videos_transporter/metrics.txt', 'w') as metrics_log:
                metrics_log.write(f"M_tracking: {M_tracking}\n")
                metrics_log.write(f"M_visual: {M_visual}\n")
                metrics_log.write(f"M_distribution: {M_distribution}\n")

            exit()
