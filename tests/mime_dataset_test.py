""" This script is used to test some functionalities of the MIME dataset loader / wrapper."""

from torch.utils.data import DataLoader

from src.data.mime import MimeHDKinectRGB

data_set = MimeHDKinectRGB(
    base_path='/home/yannik/vssil/datasets/',
    task='stir',
    start_ind=0,
    stop_ind=10,
    timesteps_per_sample=1,  # Set to -1 to return whole trajectory
    overlap=0,
    img_scale_factor=0.25
)

data_loader = DataLoader(
    data_set,
    batch_size=32,
    shuffle=True
)

print(f"Loaded data-set of {len(data_set.sample_paths)} trajectories and {len(data_loader)} samples.")

for i, sample in enumerate(data_loader):

    print(f"Sample: {i + 1}|{len(data_loader)}\t  Shape: {sample.shape}.")

    del sample

