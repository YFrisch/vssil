import torch

sample = torch.ones(size=(1, 50, 512, 512))
sample = torch.cat([sample[0:1, 0:1, :, :], sample, sample[0:1, -1:, :, :]], dim=1)
print(sample.shape)
sample = torch.cat([sample[0:1, k-1:k+2, :, :] for k in range(1, sample.shape[1] - 1)], dim=0)
print(sample.shape)
print()

sample2 = torch.ones(size=(16, 50, 512, 512))
sample2 = torch.cat([sample2[:, 0:1, :, :], sample2, sample2[:, -1:, :, :]], dim=1)
print(sample2.shape)
sample2 = torch.cat([sample2[:, k-1:k+2, :, :].unsqueeze(1) for k in range(1, sample2.shape[1] - 1)], dim=1)
print(sample2.shape)
