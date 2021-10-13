import os

import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision import transforms

from src.data.video_dataset import VideoFrameDataset, ImglistToTensor

preprocess = transforms.Compose([
        # NOTE: The first transform already converts the image range to (0, 1)
        ImglistToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

data_path = '/media/yannik/samsung_ssd/data/human_36m_processed_256pix'

data_set = VideoFrameDataset(
            root_path=data_path,
            annotationfile_path=os.path.join(data_path, 'annotations.txt'),
            num_segments=1,
            frames_per_segment=1,
            imagefile_template='img_{:05d}.jpg',
            transform=preprocess,
            random_shift=True,
            test_mode=False
)

sample, label = data_set.__getitem__(0)
sample2, label2 = data_set.__getitem__(5)

rotated_sample = torch.clone(sample)
rotated_sample2 = torch.clone(sample2)
rotated_sample = transforms.RandomRotation(degrees=90)(rotated_sample)
rotated_sample2 = transforms.RandomRotation(degrees=90)(rotated_sample2)

#sample = cv2.cvtColor(sample.squeeze(0).permute(1, 2, 0).cpu().numpy(), cv2.COLOR_BGR2GRAY)
#sample2 = cv2.cvtColor(sample2.squeeze(0).permute(1, 2, 0).cpu().numpy(), cv2.COLOR_BGR2GRAY)
#rotated_sample = cv2.cvtColor(rotated_sample.squeeze(0).permute(1, 2, 0).cpu().numpy(), cv2.COLOR_BGR2GRAY)
#rotated_sample2 = cv2.cvtColor(rotated_sample2.squeeze(0).permute(1, 2, 0).cpu().numpy(), cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures=2)
beblid = cv2.xfeatures2d.BEBLID_create(0.75)

kp1, f1 = orb.detectAndCompute(sample.squeeze(0).permute(1, 2, 0).cpu().numpy(), None)
rot_kp1, rot_f1 = orb.detectAndCompute(rotated_sample.squeeze(0).permute(1, 2, 0).cpu().numpy(), None)
kp2, f2 = orb.detectAndCompute(sample2.squeeze(0).permute(1, 2, 0).cpu().numpy(), None)
rot_kp2, rot_f2 = orb.detectAndCompute(rotated_sample2.squeeze(0).permute(1, 2, 0).cpu().numpy(), None)

print(np.linalg.norm(f1 - rot_f1, ord=2))
print(np.linalg.norm(f1 - f2, ord=2))
exit()

kp1, f1 = beblid.compute(sample.squeeze(0).permute(1, 2, 0).cpu().numpy(), kp1)
rot_kp1, rot_f1 = beblid.compute(rotated_sample.squeeze(0).permute(1, 2, 0).cpu().numpy(), rot_kp1)
kp2, f2 = beblid.compute(sample2.squeeze(0).permute(1, 2, 0).cpu().numpy(), kp2)
rot_kp2, rot_f2 = beblid.compute(rotated_sample2.squeeze(0).permute(1, 2, 0).cpu().numpy(), rot_kp2)

exit()

sample_keypts, sample_descr = beblid.detectAndCompute(sample.squeeze(0).permute(2, 1, 0).cpu().numpy(), None)
rotated_sample_keypts, rotated_sample_descr = beblid.detectAndCompute(rotated_sample.squeeze(0).permute(2, 1, 0).cpu().numpy(), None)
sample2_keypts, sample2_descr = beblid.detectAndCompute(sample2.squeeze(0).permute(2, 1, 0).cpu().numpy(), None)

print(type(sample_descr))
print(type(rotated_sample_descr))
print(type(sample2_descr))

print(f'||f(s1) - f(s1`)||: ', np.linalg.norm(x=sample_descr-rotated_sample_descr, ord=2))
print(f'||f(s1) - f(s2)||: ', np.linalg.norm(x=sample_descr-sample2_descr, ord=2))
print(f'||f(s2) - f(s1`)||: ', np.linalg.norm(x=sample2_descr-rotated_sample_descr, ord=2))

fig, ax = plt.subplots(1, 2)
ax[0].imshow(sample.squeeze(0).permute(2, 1, 0).cpu().numpy())
ax[1].imshow(rotated_sample.squeeze(0).permute(2, 1, 0).cpu().numpy())
plt.show()
