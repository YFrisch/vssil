import matplotlib.pyplot as plt
import torch
import yaml

from torch.utils.data import DataLoader

from src.agents.transporter_agent import TransporterAgent
from src.data.npz_dataset import NPZ_Dataset
from src.utils.argparse import parse_arguments
from src.utils.visualization import play_series_and_reconstruction_with_keypoints, gen_eval_imgs

if __name__ == "__main__":

    args = parse_arguments()
    args.config = '/home/yannik/vssil/results/transporter/2021_8_28_10_8/config.yml'

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

    npz_data_set = NPZ_Dataset(
        num_timesteps=200,  # 200
        root_path='/home/yannik/vssil/video_structure/testdata/acrobot_swingup_random_repeat40_00006887be28ecb8.npz',
        key_word='images'
    )

    eval_data_loader = DataLoader(
        dataset=npz_data_set,
        batch_size=1,
        shuffle=True
    )

    transporter_agent = TransporterAgent(dataset=npz_data_set, config=transporter_conf)
    transporter_agent.load_checkpoint(
        '/home/yannik/vssil/results/transporter/2021_8_28_10_8/checkpoints/chckpt_f0_e80.PTH')

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
                _sample, target = transporter_agent.preprocess(sample[:, t:t+1+t_diff, ...], label, transporter_conf)
                _sample.to(transporter_agent.device)
                target.to(transporter_agent.device)

                reconstruction = transporter_agent.model(_sample, target).clip(-0.5, 0.5)
                reconstructed_diff = (reconstruction - _sample).clip(-1.0, 1.0)
                target_diff = (target - _sample).clip(-1.0, 1.0)
                key_point_coordinates = transporter_agent.model.keypointer(_sample)[0]
                # Adapt to visualization
                key_point_coordinates[..., 1] *= -1

                samples = _sample.unsqueeze(1) if samples is None else torch.cat([samples, _sample.unsqueeze(1)], dim=1)
                reconstructions = reconstruction.unsqueeze(1) if reconstructions is None\
                    else torch.cat([reconstructions, reconstruction.unsqueeze(1)], dim=1)
                reconstructed_diffs = reconstructed_diff.unsqueeze(1) if reconstructed_diffs is None\
                    else torch.cat([reconstructed_diffs, reconstructed_diff.unsqueeze(1)], dim=1)
                key_points = key_point_coordinates.unsqueeze(1) if key_points is None \
                    else torch.cat([key_points, key_point_coordinates.unsqueeze(1)], dim=1)

                """
                fig, ax = plt.subplots(1, 5, figsize=(20, 4))
                ax[0].imshow((_sample + 0.5).squeeze().permute(1, 2, 0).cpu().numpy())
                ax[0].set_title(f"Sample t={t}")
                ax[1].imshow((target + 0.5).squeeze().permute(1, 2, 0).cpu().numpy())
                ax[1].set_title(f"Target t={t+t_diff}")
                ax[2].imshow((reconstruction + 0.5).squeeze().permute(1, 2, 0).cpu().numpy())
                ax[2].set_title(f"Reconstruction t={t+t_diff}")
                ax[3].imshow(((reconstructed_diff + 1)/2.0).squeeze().permute(1, 2, 0).cpu().numpy())
                ax[3].set_title(f"Pred. diff. t={t} to t={t+t_diff}")
                ax[4].imshow(((target_diff + 1) / 2.0).squeeze().permute(1, 2, 0).cpu().numpy())
                ax[4].set_title(f"Target. diff. t={t} to t={t + t_diff}")

                gen_eval_imgs(sample=_sample.unsqueeze(0),
                              reconstructed_diff=reconstructed_diff.unsqueeze(0),
                              key_points=key_point_coordinates.unsqueeze(0))

                plt.close()
                """

            print(samples.shape)
            print(reconstructed_diffs.shape)
            print(reconstructions.shape)
            print(key_points.shape)
            play_series_and_reconstruction_with_keypoints(image_series=samples,
                                                          reconstruction=reconstructions,
                                                          keypoint_coords=key_points)
            exit()