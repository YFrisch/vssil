import torch
import torch.nn as nn

torch.manual_seed(123)

fake_img = torch.ones((3, 8, 8))
fake_img = torch.rand_like(fake_img)
fake_img.requires_grad_(True)

lin = nn.Linear(in_features=192, out_features=50)
lin2 = nn.Linear(in_features=50, out_features=2)
print(lin.weight.requires_grad)
print(fake_img.requires_grad)

tanh = nn.Tanh()

hidden = tanh(lin(fake_img.flatten()))
fake_kpts = tanh(lin2(hidden))

print(fake_kpts)

"""
int_kpts = [
	int((-fake_kpts[0] + 1)*4),
	int((fake_kpts[1] + 1)*4) 
]
"""
int_kpts = (fake_kpts * torch.Tensor([-1, 1]) + 1) * 0.5
int_kpts = torch.floor(int_kpts * torch.tensor([8, 8])).long()

print(int_kpts)

# pixel = fake_img[:, int_kpts[0], int_kpts[1]]
pixel = torch.index_select(fake_img, 1, torch.tensor([int_kpts[0]]))
pixel = torch.index_select(pixel, 2, torch.tensor([int_kpts[1]]))

print(pixel.requires_grad)

print(pixel)

L = torch.norm(pixel, p=2)

L.backward()

print(lin.weight.grad)
