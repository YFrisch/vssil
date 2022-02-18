import yaml

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from src.utils.argparse import parse_arguments
from src.data.utils import get_dataset_from_path

from src.agents.ulosd_agent import ULOSD_Agent
from src.utils.kpt_utils import get_image_patches
from src.losses.kpt_distribution_metric import kpt_distribution_metric
from src.losses.kpt_tracking_metric import kpt_tracking_metric
from src.losses.kpt_visual_metric import kpt_visual_metric
from src.losses.spatial_consistency_loss import spatial_consistency_loss


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
    random_sampler = SubsetRandomSampler(indices=range(start_ind, stop_ind))

    eval_data_loader = DataLoader(
        dataset=data_set,
        batch_size=1,
        shuffle=False,
        sampler=random_sampler
    )

    ulosd_agent = ULOSD_Agent(dataset=data_set,
                              config=ulosd_conf)

    ulosd_agent.eval_data_loader = eval_data_loader
    ulosd_agent.load_checkpoint(
        args.checkpoint,
        map_location='cpu'
    )

    print("##### Evaluating:")
    with torch.no_grad():

        M_smooth = []
        M_distribution = []
        M_tracking = []
        M_visual = []

        for i, (sample, label) in enumerate(eval_data_loader):

            print(f"\rSample {i}|{len(range(start_ind, stop_ind))} ...", end="")

            _sample, _ = ulosd_agent.preprocess(sample, label, ulosd_conf)
            _sample.to(ulosd_agent.device)

            feature_maps, key_points = ulosd_agent.model.encode(image_sequence=_sample)

            reconstruction, gmaps = ulosd_agent.model.decode(keypoint_sequence=key_points,
                                                             first_frame=_sample[:, 0, ...].unsqueeze(1))

            patches = get_image_patches(image_sequence=sample, kpt_sequence=key_points,
                                        patch_size=(12, 12))

            M_smooth.append(spatial_consistency_loss(key_points).cpu().numpy())
            M_distribution.append(kpt_distribution_metric(key_points, img_shape=sample.shape[-2:],
                                                          n_samples=100).cpu().numpy())
            M_tracking.append(kpt_tracking_metric(key_points, sample, patch_size=(12, 12),
                                                  n_bins=20, p=float('inf'))[0].cpu().numpy())
            M_visual.append(kpt_visual_metric(key_points, sample, patch_size=(12, 12),
                                              n_bins=20, p=float('inf'))[0].cpu().numpy())

    print()
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

    with open('results/ulosd_metric.yml', 'w') as stream:
        yaml.dump(metric_dict, stream)