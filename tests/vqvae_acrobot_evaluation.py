import yaml

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.utils.argparse import parse_arguments
from src.agents.ulosd_vqvae_agent import VQVAE_Agent
from src.data.npz_dataset import NPZ_Dataset
from src.utils.visualization import play_series_and_reconstruction_with_keypoints, plot_keypoint_amplitudes

if __name__ == "__main__":

    args = parse_arguments()
    args.config = "/home/yannik/vssil/results/vqvae_acrobot/2021_11_8_19_16/config.yml"

    with open(args.config, 'r') as stream:
        vqvae_conf = yaml.safe_load(stream)
        if vqvae_conf['warm_start']:
            with open(vqvae_conf['warm_start_config'], 'r') as stream2:
                old_conf = yaml.safe_load(stream2)
                vqvae_conf['log_dir'] = old_conf['log_dir'][:-1] + "_resume/"
        else:
            vqvae_conf['log_dir'] = vqvae_conf['log_dir'] + f"/{args.id}/"
        print(vqvae_conf['log_dir'])
        vqvae_conf['multi_gpu'] = False
        vqvae_conf['device'] = 'cpu'

    npz_data_set = NPZ_Dataset(
        num_timesteps=vqvae_conf['model']['n_frames'],
        # root_path='/home/yannik/vssil/video_structure/testdata/acrobot_swingup_random_repeat40_00006887be28ecb8.npz',
        root_path=args.data,
        key_word='images',
        transform=None
    )

    vqvae_agent = VQVAE_Agent(dataset=npz_data_set,
                              config=vqvae_conf)

    eval_data_loader = DataLoader(
        dataset=npz_data_set,
        batch_size=1,
        shuffle=True
    )

    vqvae_agent.eval_data_loader = eval_data_loader
    vqvae_agent.load_checkpoint(
        "/home/yannik/vssil/results/vqvae_acrobot/2021_11_8_19_16/checkpoints/chckpt_f0_e140.PTH"
    )

    intensity_threshold = 0.5

    print("##### Evaluating:")
    with torch.no_grad():
        for i, (sample, label) in enumerate(eval_data_loader):

            sample, _ = vqvae_agent.preprocess(sample, label, vqvae_conf)
            sample.to(vqvae_agent.device)

            feature_maps = vqvae_agent.model._encoder(sample.squeeze(0))
            feature_maps = vqvae_agent.model._pre_vq_conv(feature_maps)

            key_points = vqvae_agent.model._fmap2kpt(F.softplus(feature_maps))

            gmaps = vqvae_agent.model._kpt2gmap(key_points)

            _, quantized_gmaps, _, _ = vqvae_agent.model._vq_vae(gmaps)

            _, quantized_maps, _, _ = vqvae_agent.model._vq_vae(feature_maps)

            print(quantized_maps.shape)
            print(quantized_gmaps.shape)

            stack = torch.cat([quantized_maps[0:1, ...], quantized_gmaps[0:1, ...], quantized_maps], dim=1)

            reconstruction = vqvae_agent.model._decoder(stack)

            for t in range(key_points.shape[1]):
                count = 0
                for scales in key_points[:, t, :, 2].cpu().numpy():

                    for scale in scales:

                        if scale > intensity_threshold:
                            count += 1
                print(f't: {t}\t #scales > {intensity_threshold}: {count}')

            reconstruction = vqvae_agent.model.decode(keypoint_sequence=key_points,
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
