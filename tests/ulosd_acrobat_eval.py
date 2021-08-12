import os
import yaml

import torch
from torch.utils.data import DataLoader

from src.data.npz_dataset import NPZ_Dataset
from src.agents.ulosd_agent import ULOSD_Agent
from src.utils.visualization import play_series_and_reconstruction_with_keypoints
from src.utils.argparse import parse_arguments

if __name__ == "__main__":

    data_root_path = '/home/yannik/vssil/datasets/mime_processed'
    annotations_file = os.path.join(data_root_path, 'annotations.txt')

    args = parse_arguments()
    args.config = "/home/yannik/vssil/results/ulosd/2021_8_9_16_21/config.yml"

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
        ulosd_conf['model']['n_frames'] = 4

    npz_data_set = NPZ_Dataset(
        num_timesteps=ulosd_conf['model']['n_frames'],
        root_path='/home/yannik/vssil/video_structure/testdata/acrobot_swingup_random_repeat40_00006887be28ecb8.npz',
        key_word='images'
    )

    eval_data_loader = DataLoader(
        dataset=npz_data_set,
        batch_size=1,
        shuffle=True
    )

    ulosd_agent = ULOSD_Agent(dataset=npz_data_set,
                              config=ulosd_conf)

    ulosd_agent.eval_data_loader = eval_data_loader
    ulosd_agent.load_checkpoint("/home/yannik/vssil/results/ulosd/2021_8_9_16_21/checkpoints/chckpt_f0_e90.PTH")

    print("##### Evaluating:")
    with torch.no_grad():
        for i, (sample, label) in enumerate(eval_data_loader):

            sample, _ = ulosd_agent.preprocess(sample, label, ulosd_conf)
            sample.to(ulosd_agent.device)

            feature_maps, key_points = ulosd_agent.model.encode(image_sequence=sample)

            print(key_points.shape)

            for t in range(key_points.shape[1]):
                count = 0
                for scales in key_points[:, t, :, 2].cpu().numpy():

                    for scale in scales:

                        if scale > 0.9:
                            count += 1
                print(f't: {t} #scales > 0.9: {count}')

            reconstructed_diff = ulosd_agent.model.decode(keypoint_sequence=key_points,
                                                          first_frame=sample[:, 0, ...].unsqueeze(1)).clip(-1.0, 1.0)

            play_series_and_reconstruction_with_keypoints(image_series=sample,
                                                          reconstructed_diff=reconstructed_diff,
                                                          keypoint_coords=key_points)

            if i == 0:
                exit()

    # ulosd_agent.evaluate(config=ulosd_conf)
