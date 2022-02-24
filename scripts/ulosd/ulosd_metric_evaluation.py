import os
import yaml

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset

from src.utils.argparse import parse_arguments
from src.data.utils import get_dataset_from_path

from src.agents.ulosd_agent import ULOSD_Agent
from src.utils.kpt_utils import get_image_patches, get_active_kpts
from src.utils.visualization import play_series_with_keypoints
from src.losses.kpt_distribution_metric import kpt_distribution_metric
from src.losses.kpt_tracking_metric import kpt_tracking_metric
from src.losses.kpt_visual_metric import kpt_visual_metric
from src.losses.spatial_consistency_loss import spatial_consistency_loss
from src.losses.kpt_rod_metric import kpt_rod_metric


if __name__ == "__main__":

    args = parse_arguments()

    with open(args.config, 'r') as stream:
        ulosd_conf = yaml.safe_load(stream)
        print(ulosd_conf['log_dir'])
        ulosd_conf['multi_gpu'] = False
        ulosd_conf['device'] = 'cpu'

    data_set = get_dataset_from_path(args.data, n_timesteps=30)

    # Use last 10 percent of data-set for evaluation (Not seen during training)
    stop_ind = len(data_set)
    start_ind = int(stop_ind * 0.9) + 1
    data_set = Subset(dataset=data_set, indices=range(start_ind, stop_ind))

    eval_data_loader = DataLoader(
        dataset=data_set,
        batch_size=1,
        shuffle=False,
    )

    ulosd_agent = ULOSD_Agent(dataset=data_set, config=ulosd_conf)
    ulosd_agent.eval_data_loader = eval_data_loader
    ulosd_agent.load_checkpoint(args.checkpoint, map_location='cpu')

    os.makedirs('metric_eval_results/', exist_ok=True)

    print("##### Evaluating:")
    with torch.no_grad():

        M_smooth = []
        M_distribution = []
        M_tracking = []
        M_visual = []
        M_rod = []

        for i, (sample, label) in enumerate(eval_data_loader):

            print(f"\rSample {i}|{len(range(start_ind, stop_ind))} ...", end="")

            _sample, _ = ulosd_agent.preprocess(sample, label, ulosd_conf)
            _sample.to(ulosd_agent.device)

            feature_maps, key_points = ulosd_agent.model.encode(image_sequence=_sample)

            reconstruction, gmaps = ulosd_agent.model.decode(keypoint_sequence=key_points,
                                                             first_frame=_sample[:, 0, ...].unsqueeze(1))

            # TODO: Filter for active kpts
            active_key_points = get_active_kpts(key_points)

            # Adapt key-point coordinate system
            _key_points = torch.clone(active_key_points)
            # TODO: Unify this...
            _key_points[..., :2] *= -1

            patches = get_image_patches(image_sequence=sample, kpt_sequence=key_points,
                                        patch_size=(12, 12))

            M_smooth.append(spatial_consistency_loss(key_points).cpu().numpy())
            M_distribution.append(kpt_distribution_metric(key_points, img_shape=sample.shape[-2:],
                                                          n_samples=100).cpu().numpy())
            M_tracking.append(kpt_tracking_metric(key_points, sample, patch_size=(12, 12),
                                                  n_bins=20, p=float('inf'))[0].cpu().numpy())
            M_visual.append(kpt_visual_metric(key_points, sample, patch_size=(12, 12),
                                              n_bins=20, p=float('inf'))[0].cpu().numpy())
            M_rod.append(kpt_rod_metric(key_points, sample,
                                        diameter=int(sample.shape[-1] / 10),
                                        mask_threshold=0.1))

            play_series_with_keypoints(
                image_series=sample,
                # keypoint_coords=key_points,
                keypoint_coords=_key_points,
                intensity_threshold=0.2,
                key_point_trajectory=True,
                trajectory_length=5,
                save_path=f'metric_eval_results/ulosd_sample_{i}/',
                save_frames=True
            )

            torch.save(sample, f'metric_eval_results/ulosd_sample_{i}/sample.pt')
            torch.save(feature_maps, f'metric_eval_results/ulosd_sample_{i}/fmaps.pt')
            torch.save(gmaps, f'metric_eval_results/ulosd_sample_{i}/gmaps.pt')
            torch.save(reconstruction, f'metric_eval_results/ulosd_sample_{i}/reconstruction.pt')
            torch.save(key_points, f'metric_eval_results/ulosd_sample_{i}/key_points.pt')
            torch.save(patches, f'metric_eval_results/ulosd_sample_{i}/patches.pt')

    print()
    print(M_smooth)
    print(M_distribution)
    print(M_tracking)
    print(M_visual)
    print(M_rod)

    metric_dict = {
        'smooth': M_smooth,
        'dist': M_distribution,
        'track': M_tracking,
        'visual': M_visual,
        'rod': M_rod
    }

    # Safe stuff
    with open('metric_eval_results/ulosd_metric.yml', 'w') as stream:
        yaml.dump(metric_dict, stream)
    with open('metric_eval_results/ulosd_config.yml', 'w') as stream:
        yaml.dump(ulosd_conf, stream)
    torch.save(ulosd_agent.model.state_dict(),
               'metric_eval_results/ulosd_checkpoint.PTH')
