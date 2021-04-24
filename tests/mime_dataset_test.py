from torch.utils.data import DataLoader

from data.mime_dataset_wrapper import MimeDataSet
from data.utils import play_video

data_set = MimeDataSet(
    base_path='/home/yannik/vssil/data/datasets/',
    task='stir',
    joint_data=False,
    hd_kinect_img_data=True
)

data_loader = DataLoader(
    data_set,
    batch_size=1,
    shuffle=True
)

print(f"Loaded data-set of size {len(data_loader)}")

for sample in data_loader:
    sample = sample['hd_kinect_img_series'].squeeze()
    print(f"Sample shape: ", sample.shape)
    play_video(sample)
    exit()
