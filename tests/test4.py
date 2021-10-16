import glob

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms as T
from PIL import Image

img1 = Image.open('/media/yannik/samsung_ssd/data/human_36m_processed_256pix/S1/Directions_54138969/img_00000.jpg')
img_tensor1 = T.PILToTensor()(img1).unsqueeze(0)
img2 = Image.open('/media/yannik/samsung_ssd/data/human_36m_processed_256pix/S1/Directions_54138969/img_00001.jpg')
img_tensor2 = T.PILToTensor()(img2).unsqueeze(0)
img3 = Image.open('/media/yannik/samsung_ssd/data/human_36m_processed_256pix/S1/Directions_54138969/img_00002.jpg')
img_tensor3 = T.PILToTensor()(img3).unsqueeze(0)

img_tensor = torch.cat([img_tensor1, img_tensor2, img_tensor3], dim=0).float()/255.0

print(img_tensor.shape)

N, C, H, W = img_tensor.shape

key_points = torch.zeros(size=(N, 1, 2))
key_points[..., 0] = 0.0
key_points[..., 1] = 0.0
key_points.requires_grad_(True)

grid = torch.zeros(size=(N, 50, 50, 2))

h_min = torch.zeros(size=(N, 1), dtype=torch.long)
h_max = torch.ones(size=(N, 1), dtype=torch.long) * 5
w_min = torch.zeros(size=(N, 1), dtype=torch.long)
w_max = torch.ones(size=(N, 1), dtype=torch.long) * 5

"""
patches = None
for n in range(N):
    patch = img_tensor[n:n+1, :, h_min[n]:h_max[n], w_min[n]:w_max[n]]
    patches = patch if patches is None else torch.cat([patches, patch], dim=0)

print(patches.shape)
exit()
"""

center = int(grid.shape[1]/2)
grid[:, center, center, 0] = key_points[:, 0, 0]
grid[:, center, center, 1] = key_points[:, 0, 1]
step_h = (1/H)
step_w = (1/W)

for h_i in range(grid.shape[1]):
    for w_i in range(grid.shape[2]):
        step_size_h = h_i - center
        step_size_w = w_i - center
        grid[:, h_i, w_i, 0] = key_points[:, 0, 0] + step_size_h * step_h
        grid[:, h_i, w_i, 1] = key_points[:, 0, 1] + step_size_w * step_w

patch = F.grid_sample(input=img_tensor, grid=grid, align_corners=False)

print(patch.shape)

fig, ax = plt.subplots(nrows=1, ncols=patch.shape[0])
for n in range(patch.shape[0]):
    ax[n].imshow(patch[n, ...].permute(1, 2, 0).detach().cpu().numpy())
plt.show()

L = torch.norm(patch, p=2)

L.backward()

print(key_points.grad)


