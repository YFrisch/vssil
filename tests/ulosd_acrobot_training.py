import yaml

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.utils.argparse import parse_arguments
from src.agents.ulosd_agent import ULOSD_Agent
from src.data.npz_dataset import NPZ_Dataset

if __name__ == "__main__":

    args = parse_arguments()

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

    npz_data_set = NPZ_Dataset(
        num_timesteps=ulosd_conf['model']['n_frames'],
        root_path='/home/yannik/vssil/video_structure/testdata/acrobot_swingup_random_repeat40_00006887be28ecb8.npz',
        key_word='images'
    )

    ulosd_agent = ULOSD_Agent(dataset=npz_data_set,
                              config=ulosd_conf)
    """
    data_loader = DataLoader(
        dataset=npz_data_set,
        batch_size=8,
        shuffle=True
    )

    ulosd_agent.setup(config=ulosd_conf)
    for epoch in range(100):
        rec_losses = []
        for step in range(100):
            sample, _ = next(iter(data_loader))
            sample = sample.to(ulosd_conf['device']) - 0.5
            ulosd_agent.optim.zero_grad()
            # Vision model
            feature_maps, observed_key_points = ulosd_agent.model.encode(sample)
            reconstructed_diff = ulosd_agent.model.decode(observed_key_points, sample[:, 0, ...].unsqueeze(1))

            reconstructed_diff = torch.clip(reconstructed_diff, -1.0, 1.0)

            # Dynamics model
            # TODO: Not used yet

            # Losses
            target_diff = sample - sample[:, 0, ...].unsqueeze(1)
            reconstruction_loss = ulosd_agent.loss_func(prediction=reconstructed_diff,
                                                        target=target_diff,
                                                        config=ulosd_conf)
            rec_losses.append(reconstruction_loss.detach().cpu().numpy())

            reconstruction_loss.backward()
            ulosd_agent.optim.step()
        print(f'Epoch: {epoch}\t Avg. rec. loss: {np.mean(rec_losses)}')

        fig, ax = plt.subplots(1, 4)
        ax[0].imshow(sample[0, 0, ...].permute(1, 2, 0).cpu().numpy() + 0.5)
        ax[0].set_title('sample t0')
        ax[1].imshow(sample[0, -1, ...].permute(1, 2, 0).cpu().numpy() + 0.5)
        ax[1].set_title('sample tn')
        ax[2].imshow((target_diff[0, -1, ...] + 0.5).clip(0.0, 1.0).permute(1, 2, 0).cpu().numpy())
        ax[2].set_title('diff')
        ax[3].imshow((reconstructed_diff[0, -1, ...] + 0.5).clip(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy())
        ax[3].set_title('pred. diff')
        plt.savefig(f'ep{epoch}.png')
        plt.close()
        
    exit()
    """
    ulosd_agent.train(config=ulosd_conf)
