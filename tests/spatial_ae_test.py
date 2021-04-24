from torch.utils.data import DataLoader

from models.deep_spatial_autoencoder import DeepSpatialAE
from data.mime_dataset_wrapper import MimeDataSet

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

