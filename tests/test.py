import torch
import matplotlib.pyplot as plt

a = torch.zeros(size=(3, 5, 10))
a[0, 3, 2] = 1
a[1, 1, 7] = 1
plt.figure()
plt.imshow(a.permute(1, 2, 0).cpu().numpy())
plt.scatter(x=2, y=7, color='blue')
plt.show()

