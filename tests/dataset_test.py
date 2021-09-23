import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from src.data.npz_dataset import NPZ_Dataset


if __name__ == "__main__":

    tran = transforms.RandomApply([
        transforms.ColorJitter(brightness=.5, hue=.3),
        transforms.RandomAdjustSharpness(sharpness_factor=2.0),
        #transforms.RandomEqualize(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5)
    ])

    npz_data_set = NPZ_Dataset(
        num_timesteps=16,
        #root_path='/home/yannik/vssil/video_structure/testdata/acrobot_swingup_random_repeat40_00006887be28ecb8.npz',
        root_path='/media/yannik/samsung_ssd/data/npz_data/pong.npz',
        key_word='images',
        transform=tran
    )

    sample = next(iter(npz_data_set))[0]

    plt.figure()
    plt.imshow(sample[0, ...].permute(1, 2, 0).cpu().numpy())
    plt.show()
