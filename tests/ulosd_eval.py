import yaml
import argparse

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.data.utils import combine_mime_hd_kinect_tasks
from src.agents.ulosd_agent import ULOSD_Agent
from src.losses.temporal_separation_loss import temporal_separation_loss

parser = argparse.ArgumentParser()
parser.add_argument('config_path', metavar='CP', type=str, help='Path to config file.')
parser.add_argument('data_path', metavar='DP', type=str, help='Base path to dataset.')
args = parser.parse_args()

with open(args.config_path, 'r') as stream:
    ulosd_conf = yaml.safe_load(stream)
    ulosd_conf['device'] = 'cpu'
    ulosd_conf['data']['tasks'] = ['pull_one_hand']

data_set = combine_mime_hd_kinect_tasks(
    task_list=ulosd_conf['data']['tasks'],
    base_path=args.data_path,
    timesteps_per_sample=ulosd_conf['model']['n_frames'],
    overlap=ulosd_conf['data']['overlap'],
    img_shape=eval(ulosd_conf['data']['img_shape'])
)

eval_data_loader = DataLoader(
    dataset=data_set,
    batch_size=1,
    shuffle=True
)

ulosd_agent = ULOSD_Agent(dataset=data_set,
                          config=ulosd_conf)

ulosd_agent.eval_data_loader = eval_data_loader

with torch.no_grad():
    for i, sample in enumerate(eval_data_loader):
        sample, _ = ulosd_agent.preprocess(sample, ulosd_conf)

        feature_maps, key_points = ulosd_agent.model.encode(image_sequence=sample)

        reconstruction = ulosd_agent.model(sample)
        print(reconstruction.min())
        print(reconstruction.max())

        rec_loss = ulosd_agent.loss_func(prediction=reconstruction, target=sample)

        sep_loss = ulosd_agent.separation_loss(keypoint_coordinates=key_points, config=ulosd_conf)

        l1_penalty = ulosd_agent.l1_activation_penalty(feature_maps=feature_maps, config=ulosd_conf)

        l2_kernel_reg = ulosd_agent.l2_kernel_regularization(config=ulosd_conf)

        print(f"Sample {i}\t Rec. loss: {rec_loss}\t Sep. loss: {sep_loss}\t L1 penalty: {l1_penalty}\t "
              f"L2 reg: {l2_kernel_reg}")

        np_sample = sample[0, 0, ...].cpu().numpy().transpose(1, 2, 0) + 0.5
        np_rec = reconstruction[0, 0, ...].cpu().numpy().transpose(1, 2, 0) + 0.5
        fig, ax = plt.subplots(2, 1)
        ax[0].imshow(np_sample)
        ax[1].imshow(np_rec)
        plt.show()
        if i == 5:
            exit()

# ulosd_agent.evaluate(config=ulosd_conf)
