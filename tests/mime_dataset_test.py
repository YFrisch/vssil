from torch.utils.data import DataLoader

from data.mime_dataset_wrapper import MimeDataSet

data_loader = DataLoader(MimeDataSet())

for sample in data_loader:
    print(sample.shape)
