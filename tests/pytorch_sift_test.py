import os

import matplotlib.pyplot as plt
import albumentations as A
import torch
import torchvision.transforms
from torchvision import transforms

from src.data.video_dataset import VideoFrameDataset, ImglistToTensor
from src.utils.pytorch_sift import SIFTNet

preprocess = transforms.Compose([
        # NOTE: The first transform already converts the image range to (0, 1)
        ImglistToTensor(),
        transforms.CenterCrop(size=200)
        # transforms.RandomCrop(size=16)
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

data_path = '/media/yannik/samsung_ssd/data/human_36m_processed_256pix'

data_set = VideoFrameDataset(
            root_path=data_path,
            annotationfile_path=os.path.join(data_path, 'annotations.txt'),
            num_segments=1,
            frames_per_segment=10,
            imagefile_template='img_{:05d}.jpg',
            transform=preprocess,
            random_shift=True,
            test_mode=False
)

sample, label = data_set.__getitem__(0)
sample = transforms.Grayscale()(sample)

#rotated_sample = transforms.RandomRotation(
#    degrees=45, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, fill=0.5)(sample)

#rotated_sample = torch.from_numpy(A.RandomRotate90(always_apply=True)(image=sample)['image'])

sample2, label2 = data_set.__getitem__(10)
sample2 = transforms.Grayscale()(sample2)

#rotated_sample2 = torch.from_numpy(A.RandomRotate90(always_apply=True)(image=sample2)['image'])

#rotated_sample2 = transforms.RandomRotation(
#    degrees=45, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, fill=0.5)(sample2)

sift_ex = SIFTNet(
    patch_size=200,
    sigma_type='hesamp',
    mask_type='CircularGauss'
)

print(sample.shape)

sample_t0 = sift_ex(sample[0:1, ...])
sample_tn = sift_ex(sample[-2:-1, ...])
sample_tn_rotated = sift_ex(sample[-2:-1, ...].permute(0, 1, 3, 2))

sample2_t0 = sift_ex(sample2[0:1, ...])
sample2_tn = sift_ex(sample2[-2:-1, ...])
sample2_tn_rotated = sift_ex(sample2[-2:-1, ...].permute(0, 1, 3, 2))

print('s1 tn - s1 t0:', torch.norm(input=(sample_tn - sample_t0), p=2).item())
print('s1 tn - s1 tn rotated:', torch.norm(input=(sample_tn - sample_tn_rotated), p=2).item())
print('s2 tn - s2 t0:', torch.norm(input=(sample2_tn - sample2_t0), p=2).item())
print('s2 tn - s2 tn rotated:', torch.norm(input=(sample2_tn - sample2_tn_rotated), p=2).item())
print('s1 tn - s2 t0:', torch.norm(input=(sample_tn - sample2_t0), p=2).item())

fig, ax = plt.subplots(2, 3)
ax[0, 0].imshow(sample[0:1, ...].squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap='gray')
ax[0, 0].set_title('sample1 t=0')

ax[0, 1].imshow(sample[-2:-1, ...].squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap='gray')
ax[0, 1].set_title('sample1 t=n')

ax[0, 2].imshow(sample[-2:-1, ...].squeeze(0).permute(2, 1, 0).cpu().numpy(), cmap='gray')
ax[0, 2].set_title('sample1 t=n rotated')

ax[1, 0].imshow(sample2[0:1, ...].squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap='gray')
ax[1, 0].set_title('sample2 t=0')

ax[1, 1].imshow(sample2[-2:-1, ...].squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap='gray')
ax[1, 1].set_title('sample2 t=n')

ax[1, 2].imshow(sample2[-2:-1, ...].squeeze(0).permute(2, 1, 0).cpu().numpy(), cmap='gray')
ax[1, 2].set_title('sample2 t=n rotated')

plt.show()
