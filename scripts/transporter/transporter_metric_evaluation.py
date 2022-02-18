import yaml

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from src.agents.transporter_agent import TransporterAgent
from src.data.utils import get_dataset_from_path
from src.utils.argparse import parse_arguments
from src.utils.kpt_utils import get_image_patches
from src.losses.kpt_distribution_metric import kpt_distribution_metric
from src.losses.kpt_visual_metric import kpt_visual_metric
from src.losses.kpt_tracking_metric import kpt_tracking_metric
from src.losses.spatial_consistency_loss import spatial_consistency_loss


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
    random_sampler = SubsetRandomSampler(indices=range(start_ind, stop_ind))

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

        samples = None
        patches = None
        kpts = None

        M_smooth = []
        M_distribution = []
        M_tracking = []
        M_visual = []

        for i, (sample, label) in enumerate(eval_data_loader):

            samples = None
            reconstructed_diffs = None
            reconstructions = None
            key_points = None

            t_diff = 1

            print()
            for t in range(sample.shape[1] - t_diff):
                print(f'\r{t}|{sample.shape[1] - t_diff}', end="")
                _sample, target = transporter_agent.preprocess(
                    sample[:, t:t + 1 + t_diff, ...], label, transporter_conf)
                _sample.to(transporter_agent.device)
                target.to(transporter_agent.device)

                reconstruction = transporter_agent.model(_sample, target).clip(-0.5, 0.5)
                reconstructed_diff = (reconstruction - _sample).clip(-1.0, 1.0)
                target_diff = (target - _sample).clip(-1.0, 1.0)
                key_point_coordinates = transporter_agent.model.keypointer(_sample)[0]

                samples = sample[:, t:t+1, ...] if samples is None \
                    else torch.cat([samples, sample[:, t:t+1, ...]], dim=1)
                reconstructions = reconstruction.unsqueeze(1) if reconstructions is None \
                    else torch.cat([reconstructions, reconstruction.unsqueeze(1)], dim=1)
                key_points = key_point_coordinates.unsqueeze(1) if key_points is None \
                    else torch.cat([key_points, key_point_coordinates.unsqueeze(1)], dim=1)
            print()

            patches = get_image_patches(image_sequence=samples, kpt_sequence=key_points,
                                        patch_size=(12, 12))

            M_smooth.append(spatial_consistency_loss(key_points).cpu().numpy())
            M_distribution.append(kpt_distribution_metric(key_points, img_shape=samples.shape[-2:],
                                                          n_samples=100).cpu().numpy())
            M_tracking.append(kpt_tracking_metric(key_points, samples, patch_size=(12, 12),
                                                  n_bins=20, p=float('inf'))[0].cpu().numpy())
            M_visual.append(kpt_visual_metric(key_points, samples, patch_size=(12, 12),
                                              n_bins=20, p=float('inf'))[0].cpu().numpy())

    print(M_smooth)
    print(M_distribution)
    print(M_tracking)
    print(M_visual)

    metric_dict = {
        'smooth': M_smooth,
        'dist': M_distribution,
        'track': M_tracking,
        'visual': M_visual
    }

    with open('results/transporter_metric.yml', 'w') as stream:
        yaml.dump(metric_dict, stream)
