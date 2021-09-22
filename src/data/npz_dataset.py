import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class NPZ_Dataset(Dataset):
    """ Dataset class with data loaded from a .npz file."""

    def __init__(self,
                 num_timesteps: int,
                 root_path: str,
                 key_word: str,
                 transform=None):
        super(NPZ_Dataset, self).__init__()

        self.data = torch.tensor(read_npz_data(root_path)[key_word])
        self.num_timesteps = num_timesteps
        self.transform = transform

    def __len__(self):
        return len(self.data) - self.num_timesteps

    def __getitem__(self, index):

        float_img_tensor = torch.FloatTensor(self.data[index:index+self.num_timesteps]/255)
        if self.transform is not None:
            float_img_tensor = self.transform(float_img_tensor)
        return float_img_tensor, torch.empty([])


def read_npz_data(path: str):
    npz_file = np.load(path)

    np_images = npz_file['image'].transpose(0, 3, 1, 2)
    np_actions = npz_file['action']
    np_rewards = npz_file['reward']
    np_orientations = npz_file['orientations']
    np_velocities = npz_file['velocity']
    return {
        'images': np_images,
        'actions': np_actions,
        'rewards': np_rewards,
        'orientations': np_orientations,
        'velocities': np_velocities
    }


if __name__ == "__main__":
    npz_data_set = NPZ_Dataset(
        num_timesteps=16,
        root_path='/home/yannik/vssil/video_structure/testdata/'
                  'acrobot_swingup_random_repeat40_00006887be28ecb8.npz',
        key_word='images')

    data_loader = DataLoader(dataset=npz_data_set, batch_size=4, shuffle=True)
    print(f'Read {len(npz_data_set)} samples.')
    for i, (sample, label) in enumerate(data_loader):
        sample -= 0.5
        print(sample.mean())
        print(sample.max())
        print(sample.min())
        print(sample.shape)
        #numpy_to_mp4(img_array=sample.squeeze(0).permute(0, 2, 3, 1).cpu().numpy(),
        #             target_path='acrobot_npz_test.avi')
        exit()
