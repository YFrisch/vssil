import torch
import torch.nn.functional as F

# softmax = torch.tensor([0.992, 0.004, 0.002])
pred= torch.tensor([1.0, 0.1, 3.0])
softmax = F.softmax(pred, dim=0)
print(softmax)
log_softmax = torch.log(softmax)
target = torch.tensor([0.0, 0.0, 1.0])

ce = F.binary_cross_entropy_with_logits(log_softmax, target,
                                        reduction='none',
                                        weight=torch.tensor([0.3, 0.0, 0.7]))

L = ce.sum()

print(L)

target_classes = torch.argmax(target, dim=0).unsqueeze(0)
ce2 = F.cross_entropy(input=pred.unsqueeze(0), target=target_classes, weight=torch.tensor([0.3, 0.0, 0.7]))

print(ce2)
