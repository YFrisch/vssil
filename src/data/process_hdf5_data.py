import h5py
import matplotlib.pyplot as plt


f = h5py.File("/media/yannik/samsung_ssd/data/hdf5_data/image.hdf5", 'r')

# /data/demo_X/obs/agentview_image

n_samples = len(f['data'].keys())
print(n_samples)

img = f['data']['demo_0']['obs']['agentview_image']

print(img)

fig, ax = plt.subplots(1, 1)
for t in range(img.shape[0]):
    ax.imshow(img[t])
plt.show()



