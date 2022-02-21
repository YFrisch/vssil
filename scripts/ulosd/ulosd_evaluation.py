import yaml

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from src.utils.argparse import parse_arguments
from src.data.utils import get_dataset_from_path
from src.agents.ulosd_agent import ULOSD_Agent
from src.utils.visualization import play_series_with_keypoints, plot_keypoint_amplitudes


if __name__ == "__main__":

    args = parse_arguments()

    with open(args.config, 'r') as stream:
        ulosd_conf = yaml.safe_load(stream)
        print(ulosd_conf['log_dir'])
        ulosd_conf['multi_gpu'] = False
        ulosd_conf['device'] = 'cpu'

    data_set = get_dataset_from_path(args.data, n_timesteps=150)

    # Use last 10 percent of data-set for evaluation (Not seen during training)
    stop_ind = len(data_set)
    start_ind = int(stop_ind * 0.9) + 1
    # random_sampler = SubsetRandomSampler(indices=range(start_ind, stop_ind))
    random_sampler = SubsetRandomSampler(indices=[stop_ind - 4])  # Only single sample

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

    intensity_threshold = 0.5

    print("##### Evaluating:")
    with torch.no_grad():
        for i, (sample, label) in enumerate(eval_data_loader):

            _sample, _ = ulosd_agent.preprocess(sample, label, ulosd_conf)
            _sample.to(ulosd_agent.device)

            feature_maps, key_points = ulosd_agent.model.encode(image_sequence=_sample)

            for t in range(key_points.shape[1]):
                count = 0
                for scales in key_points[:, t, :, 2].cpu().numpy():
                    for scale in scales:
                        if scale > intensity_threshold:
                            count += 1
                print(f't: {t}\t #scales > {intensity_threshold}: {count}')

            reconstruction, gmaps = ulosd_agent.model.decode(keypoint_sequence=key_points,
                                                             first_frame=_sample[:, 0, ...].unsqueeze(1))

            # Adapt for visualization
            _key_points = torch.clone(key_points)
            _key_points[..., :2] *= -1

            play_series_with_keypoints(image_series=sample,
                                       keypoint_coords=_key_points,
                                       intensity_threshold=intensity_threshold,
                                       key_point_trajectory=True,
                                       trajectory_length=5,
                                       save_path='./result_videos_ulosd/',
                                       save_frames=True)

            plot_keypoint_amplitudes(keypoint_coordinates=key_points,
                                     intensity_threshold=intensity_threshold,
                                     target_path='./result_videos_ulosd/')

            if i == 0:
                exit()
