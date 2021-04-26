from torch.utils.data import DataLoader

from data.mime_dataset_wrapper import MimeDataSet
from data.utils import play_video

data_set = MimeDataSet(
    base_path='/home/yannik/vssil/data/datasets/',
    task='stir',
    joint_data=False,
    hd_kinect_img_data=True,
    rd_kinect_img_data=False,
    img_scale_factor=0.3,
    timesteps_per_sample=10
)

data_loader = DataLoader(
    data_set,
    batch_size=10,
    shuffle=True
)

print(f"Loaded data-set of {len(data_set.trajectory_lengths)} trajectories and {len(data_loader)} samples.")

for sample in data_loader:
    # sample = sample['rd_kinect_img_series'].squeeze()
    sample = sample['hd_kinect_img_series']
    print(f"Sample shape: ", sample.shape)
    play_video(sample[0, ...].squeeze())
    exit()
