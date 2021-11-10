import torch
import matplotlib.pyplot as plt

from src.models.vqvae import KeyPointsToGaussianMaps

N, T, K, D = 8, 1, 1, 4

inp = torch.randint(low=0, high=16, size=(N, T, K, D)).float()  # (N, T, K, 4)

fake_net = torch.nn.Sequential(
    torch.nn.Linear(in_features=inp.shape[-1], out_features=inp.shape[-1]),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(in_features=inp.shape[-1], out_features=inp.shape[-1]),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(in_features=inp.shape[-1], out_features=inp.shape[-1])
)

outp = fake_net(inp).view((N, T, K, D))

kpts2gaussian = KeyPointsToGaussianMaps(
    batch_size=N, time_steps=T,
    n_kpts=K, heatmap_width=16
)

x, y = kpts2gaussian.get_grid()

ref = torch.randint(low=0, high=16, size=(N, T, K, D)).float()
ref_map = kpts2gaussian(ref)

fmap = kpts2gaussian(outp)

L = torch.norm(fmap - ref_map, p=1)
L.backward()

print(fmap.shape)
print(ref_map.shape)

print(fake_net[0].weight.grad)

for n in range(fmap.shape[0]):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].pcolormesh(x[n, 0, 0, ...], y[n, 0, 0, ...], fmap[n, 0, 0, ...].detach(), shading='auto')
    ax[1].pcolormesh(x[n, 0, 0, ...], y[n, 0, 0, ...], ref_map[n, 0, 0, ...].detach(), shading='auto')
    #fig.colorbar()
    plt.show()





