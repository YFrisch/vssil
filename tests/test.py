import torch
import matplotlib.pyplot as plt

a = torch.zeros(size=(3, 5, 10))
a[0, 3, 2] = 1
a[1, 1, 7] = 1
plt.figure()
plt.imshow(a.permute(1, 2, 0).cpu().numpy())
plt.scatter(7, 2, color='blue')  # (x=width=7, y=height=2)
plt.show()

