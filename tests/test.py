import torch
from src.models.utils import activation_dict

prelu_func = activation_dict['prelu'](init=0.25)
for n, p in prelu_func.named_parameters():
    print(n)
prelu_func2 = activation_dict['prelu'](init=0.5)
for n, p in prelu_func2.named_parameters():
    print(n)
x = torch.ones((1, 3, 3))
x[0, 1, 1] = -1
y = prelu_func(x)
y2 = prelu_func2(x)
print(y)
print(y2)
