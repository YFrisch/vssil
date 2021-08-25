import os

import mat73
from scipy.io import loadmat

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class Human80K_DataSet(Dataset):

    def __init__(self,
                 activity_id: str,
                 root_path: str = '/media/yannik/MyDrive/Data/Human80K/',):

        assert os.path.isdir(root_path)
        self.root_path = root_path

        mat_file_path = os.path.join(root_path, f'ActivitySpecific_{activity_id}.mat')
        assert os.path.isfile(mat_file_path), \
            f'{mat_file_path} not found.'

        try:
            print('##### Loading .mat file:')
            # self.scipy_mat_data = loadmat(mat_file_path)
            self.data_dict = mat73.loadmat(mat_file_path)
        except Exception as e:
            print(f'Could not load {mat_file_path}!')
            print(e)
            exit()

        print(type(self.data_dict))

    def load_mat_data(self, path: str):
        return loadmat(path)

    def __getitem__(self, index) -> T_co:
        pass


if __name__ == "__main__":

    data_set = Human80K_DataSet(activity_id='04')


