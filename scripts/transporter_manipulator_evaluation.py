import os
import yaml

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from src.agents.transporter_agent import TransporterAgent
from src.data.video_dataset import VideoFrameDataset, ImglistToTensor
from src.utils.argparse import parse_arguments
from src.utils.visualization import play_series_and_reconstruction_with_keypoints, play_series_with_keypoints,\
    plot_keypoint_amplitudes
from src.losses.kpt_metrics import get_image_patches, tracking_metric, visual_difference_metric, distribution_metric


if __name__ == "__main__":

    args = parse_arguments()
    # NOTE: Edit config here
    args.config = "/home/yannik/vssil/results/transporter_manipulator/2022_1_7_20_11/config.yml"

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
        transporter_conf['model']['n_frames'] = 2

    # Apply any number of torchvision transforms here as pre-processing
    preprocess = transforms.Compose([
        # NOTE: The first transform already converts the image range to (0, 1)
        ImglistToTensor(),

    ])

    data_set = VideoFrameDataset(
        root_path=args.data,
        annotationfile_path=os.path.join(args.data, 'annotations.txt'),
        num_segments=1,
        frames_per_segment=150,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess,
        random_shift=False,
        test_mode=True
    )

    data_set = Subset(data_set, indices=[100])

    eval_data_loader = DataLoader(
        dataset=data_set,
        batch_size=1,
        # shuffle=False
        shuffle=True
    )

    transporter_agent = TransporterAgent(dataset=data_set, config=transporter_conf)
    transporter_agent.eval_data_loader = eval_data_loader
    # NOTE: Edit checkpoint here
    transporter_agent.load_checkpoint(
        "/home/yannik/vssil/results/transporter_manipulator/2022_1_7_20_11/checkpoints/chckpt_f0_e135.PTH"
    )

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

                #samples = _sample.unsqueeze(1) if samples is None else torch.cat([samples, _sample.unsqueeze(1)], dim=1)
                samples = sample[:, t:t+1, ...] if samples is None \
                    else torch.cat([samples, sample[:, t:t+1, ...]], dim=1)
                reconstructions = reconstruction.unsqueeze(1) if reconstructions is None \
                    else torch.cat([reconstructions, reconstruction.unsqueeze(1)], dim=1)
                key_points = key_point_coordinates.unsqueeze(1) if key_points is None \
                    else torch.cat([key_points, key_point_coordinates.unsqueeze(1)], dim=1)
            
            play_series_with_keypoints(image_series=samples,
                                       keypoint_coords=key_points,
                                       key_point_trajectory=True)
            plot_keypoint_amplitudes(keypoint_coordinates=key_points,
                                     target_path=".")

            patches = get_image_patches(image_sequence=samples, kpt_sequence=key_points,
                                        patch_size=(16, 16))

            M_tracking = tracking_metric(patches)
            M_visual = visual_difference_metric(patches)
            M_distribution = distribution_metric(key_points, (16, 16))

            with open('metrics.txt', 'w') as metrics_log:
                metrics_log.write(f"M_tracking: {M_tracking}\n")
                metrics_log.write(f"M_visual: {M_visual}\n")
                metrics_log.write(f"M_distribution: {M_distribution}\n")

            exit()