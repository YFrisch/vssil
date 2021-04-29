""" This script is used to test some functionalities of the MIME dataset loader / wrapper."""

from torch.utils.data import DataLoader

from src.data.mime_dataset_wrapper import MimeDataSet
from src.data.utils import play_video

data_set = MimeDataSet(
    base_path='/home/yannik/vssil/data/datasets/',
    task='stir',
    start_ind=0,
    stop_ind=10,
    joint_data=False,
    hd_kinect_img_data=True,
    rd_kinect_img_data=False,
    img_scale_factor=1.0,
    timesteps_per_sample=20  # Set to -1 to return whole trajectory
)

data_loader = DataLoader(
    data_set,
    batch_size=8,
    shuffle=True
)

print(f"Loaded data-set of {len(data_set.trajectory_lengths)} trajectories and {len(data_loader)} samples.")

for i, sample in enumerate(data_loader):
    # sample = sample['rd_kinect_img_series'].squeeze()
    sample = sample['hd_kinect_img_series']
    print(f"Sample: {i}|{len(data_loader)}\t  Shape: ", tuple(sample.shape))
    # play_video(sample[0, ...].squeeze())
    del sample

