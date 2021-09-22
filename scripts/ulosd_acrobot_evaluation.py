import yaml

import torch
from torch.utils.data import DataLoader

from src.data.npz_dataset import NPZ_Dataset
from src.agents.ulosd_agent import ULOSD_Agent
from src.utils.visualization import play_series_and_reconstruction_with_keypoints, numpy_to_mp4
from src.utils.argparse import parse_arguments

if __name__ == "__main__":

    args = parse_arguments()
    # NOTE: Change config of your checkpoint here:
    args.config = "/home/yannik/vssil/results/ulosd_acrobat_tc_triplet/2021_9_21_15_55/config.yml"

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

    npz_data_set = NPZ_Dataset(
        num_timesteps=200,
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
    # NOTE: Change checkpoint to evaluate here:
    ulosd_agent.load_checkpoint("/home/yannik/vssil/results/ulosd_acrobat_tc_triplet/2021_9_21_15_55/checkpoints/chckpt_f0_e69.PTH")

    intensity_threshold = 0.7

    print("##### Evaluating:")
    with torch.no_grad():
        for i, (sample, label) in enumerate(eval_data_loader):

            sample, _ = ulosd_agent.preprocess(sample, label, ulosd_conf)
            sample.to(ulosd_agent.device)

            """
            numpy_to_mp4(
                img_array=sample[0, ...].permute(0, 2, 3, 1).cpu().numpy(),
                target_path='../pytorch_sample_test.avi'
            )

            exit()
            """

            feature_maps, key_points = ulosd_agent.model.encode(image_sequence=sample)

            for t in range(key_points.shape[1]):
                count = 0
                for scales in key_points[:, t, :, 2].cpu().numpy():

                    for scale in scales:

                        if scale > intensity_threshold:
                            count += 1
                print(f't: {t}\t #scales > {intensity_threshold}: {count}')

            reconstruction = ulosd_agent.model.decode(keypoint_sequence=key_points,
                                                      first_frame=sample[:, 0, ...].unsqueeze(1))

            play_series_and_reconstruction_with_keypoints(image_series=sample,
                                                          reconstruction=reconstruction,
                                                          keypoint_coords=key_points,
                                                          intensity_threshold=intensity_threshold,
                                                          key_point_trajectory=True,
                                                          trajectory_length=20)

            if i == 0:
                exit()

    # ulosd_agent.evaluate(config=ulosd_conf)
