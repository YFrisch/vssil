import os.path

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.video_dataset import VideoFrameDataset, ImglistToTensor

if __name__ == "__main__":
    # data_root_path = '/media/yannik/samsung_ssd/data/mime_processed_256pix'
    # data_root_path = '/media/yannik/samsung_ssd/data/ball_processed_64pix'
    data_root_path = '/media/yannik/samsung_ssd/data/simitate_processed_128pix'
    # data_root_path = '/media/yannik/samsung_ssd/data/human_36m_processed_128pix'
    annotations_file = os.path.join(data_root_path, 'annotations.txt')

    preprocess = transforms.Compose([
        # NOTE: The first transform already converts the range to (0, 1)
        ImglistToTensor(),
        transforms.RandomChoice([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=90)], p=0.3),
            # transforms.RandomApply([transforms.ColorJitter(brightness=.5, hue=.3)], p=0.3)
        ])
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_set = VideoFrameDataset(
        root_path=data_root_path,
        annotationfile_path=annotations_file,
        num_segments=1,
        frames_per_segment=8,
        imagefile_template='img_{:05d}.jpg',
        transform=preprocess,
        random_shift=True,
        test_mode=True
    )

    data_loader = DataLoader(
        dataset=data_set,
        batch_size=1,
        shuffle=True
    )

    print(len(data_set))

    time_diff = 4
    for i, sample in enumerate(data_loader):
        frames, labels = sample
        t0_frame = frames[0, 0, ...].permute(1, 2, 0).cpu().numpy()
        tn_frame = frames[0, time_diff, ...].permute(1, 2, 0).cpu().numpy()
        print(frames.shape)
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(t0_frame)
        ax[0].set_title('t = 0')
        ax[1].imshow(tn_frame)
        ax[1].set_title(f't = {time_diff}')
        ax[2].imshow(((tn_frame-t0_frame) + 0.5).clip(0.0, 1.0))
        ax[2].set_title(f'difference')
        plt.show()
        exit()
        print(f"{i} {list(labels)}")
        if i == 500:
            break
