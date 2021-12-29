import os
import yaml

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.utils.argparse import parse_arguments
from src.agents.ulosd_agent import ULOSD_Agent
from src.data.video_dataset import VideoFrameDataset, ImglistToTensor
from src.utils.visualization import play_series_and_reconstruction_with_keypoints, plot_keypoint_amplitudes

if __name__ == "__main__":

    args = parse_arguments()

    # Change config for evaluation here
    args.config = "/home/yannik/vssil/results/ulosd_manipulator/2021_12_18_12_57/config.yml"

    with open(args.config, 'r') as stream:
        ulosd_conf = yaml.safe_load(stream)
        if ulosd_conf['warm_start']:
            with open(ulosd_conf['warm_start_config'], 'r') as stream2:
                old_conf = yaml.safe_load(stream2)
                ulosd_conf['log_dir'] = old_conf['log_dir'][:-1] + "_resume/"
        else:
            ulosd_conf['log_dir'] = ulosd_conf['log_dir'] + f"/{args.id}/"
        print(ulosd_conf['log_dir'])
        ulosd_conf['multi_gpu'] = False
        ulosd_conf['device'] = 'cpu'

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

    data_set = Subset(data_set, [107])

    eval_data_loader = DataLoader(
        dataset=data_set,
        batch_size=1,
        shuffle=False
    )

    ulosd_agent = ULOSD_Agent(dataset=data_set,
                              config=ulosd_conf)

    ulosd_agent.eval_data_loader = eval_data_loader
    # Change checkpoint for evaluation here
    ulosd_agent.load_checkpoint(
        "/home/yannik/vssil/results/ulosd_manipulator/2021_12_18_12_57/checkpoints/chckpt_f0_e280.PTH",
        map_location='cpu'
    )

    intensity_threshold = 0.4

    print("##### Evaluating:")
    with torch.no_grad():
        for i, (sample, label) in enumerate(eval_data_loader):

            sample, _ = ulosd_agent.preprocess(sample, label, ulosd_conf)
            sample.to(ulosd_agent.device)

            feature_maps, key_points = ulosd_agent.model.encode(image_sequence=sample)

            for t in range(key_points.shape[1]):
                count = 0
                for scales in key_points[:, t, :, 2].cpu().numpy():
                    for scale in scales:
                        if scale > intensity_threshold:
                            count += 1
                print(f't: {t}\t #scales > {intensity_threshold}: {count}')

            reconstruction, gmaps = ulosd_agent.model.decode(keypoint_sequence=key_points,
                                                             first_frame=sample[:, 0, ...].unsqueeze(1))

            play_series_and_reconstruction_with_keypoints(image_series=sample,
                                                          reconstruction=reconstruction,
                                                          keypoint_coords=key_points,
                                                          feature_maps=feature_maps,
                                                          intensity_threshold=intensity_threshold,
                                                          key_point_trajectory=True,
                                                          trajectory_length=20)

            plot_keypoint_amplitudes(keypoint_coordinates=key_points,
                                     intensity_threshold=intensity_threshold,
                                     target_path='/home/yannik/vssil')

            if i == 0:
                exit()
