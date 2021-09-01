import os

import mat73
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
import matplotlib.pyplot as plt


def convert_mat(root_path: str, activity_id: str, skip_existing: bool = True):
    """ Reads in the .mat file containing a dict of rgb numpy arrays,
        then saves them as individual .npy files

        NOTE: This method requires a lot of RAM!
    """

    mat_file_path = os.path.join(root_path, f'ActivitySpecific_{activity_id}.mat')
    assert os.path.isfile(mat_file_path), \
        f'{mat_file_path} not found.'

    try:
        print('##### Loading .mat file')
        # self.scipy_mat_data = loadmat(mat_file_path)
        data_dict = mat73.loadmat(mat_file_path)
    except Exception as e:
        print(f'Could not load {mat_file_path}!')
        print(e)
        exit()

    os.makedirs(os.path.join(root_path, 'test/'), exist_ok=True)
    os.makedirs(os.path.join(root_path, 'train/'), exist_ok=True)

    print('##### Saving training .npy samples')
    for i, img_list in enumerate(data_dict['Ftrain']):
        print(f"{i}|{len(data_dict['Ftrain'])}\r", end="")
        np_img_array = img_list[0]
        np_img_path = os.path.join(root_path, f'train/act{activity_id}_img{i}')
        if skip_existing and os.path.isfile(np_img_path):
            continue
        else:
            np.save(file=np_img_path, arr=np_img_array)
        del np_img_array

    print('##### Saving testing .npy samples')
    for i, img_list in enumerate(data_dict['Ftest']):
        print(f"{i}|{len(data_dict['Ftest'])}\r", end="")
        np_img_array = img_list[0]
        np_img_path = os.path.join(root_path, f'test/act{activity_id}_img{i}')
        if skip_existing and os.path.isfile(np_img_path):
            continue
        else:
            np.save(file=np_img_path, arr=np_img_array)
        del np_img_array



class Human80K_DataSet(Dataset):

    def __init__(self,
                 activity_id: str,
                 mode: str = 'train',
                 root_path: str = '/media/yannik/MyDrive/Data/Human80K/',):

        self.activity_id = activity_id
        assert os.path.isdir(root_path)
        self.root_path = root_path

        self.file_paths = self.read_file_paths(mode)

        if len(self.file_paths) == 0:
            raise ValueError("No data found.")

    def read_file_paths(self, mode: str):
        file_paths = []
        files_path = os.path.join(self.root_path + f'{mode}/')
        assert os.path.isdir(files_path)
        for file_name in os.listdir(files_path):
            if file_name.startswith(f'act{self.activity_id}'):
                file_paths.append(os.path.join(files_path, file_name))
            else:
                continue
        return file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index) -> T_co:

        # Import .npy file as numpy array
        file_path = self.file_paths[index]
        numpy_array = np.load(file_path)

        # Convert to torch
        torch_tensor = torch.from_numpy(numpy_array)

        return torch_tensor


if __name__ == "__main__":

    convert_mat(root_path='/media/yannik/MyDrive/Data/Human80K/', activity_id='06')

    exit()

    data_set = Human80K_DataSet(activity_id='04')

    data_loader = DataLoader(
        dataset=data_set,
        batch_size=1,
        shuffle=True
    )

    sample = next(iter(data_loader))
    print(sample.shape)


