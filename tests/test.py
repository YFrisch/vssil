import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co


class SimpleDataSet(Dataset):

    def __init__(self):
        super(SimpleDataSet, self).__init__()

        self.data = np.arange(1, 11)

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    data_set = SimpleDataSet()
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=1,
        shuffle=True
    )
    n_iterations = 5
    for epoch in range(4):
        generator = iter(data_loader)
        for iteration in range(n_iterations):
            try:
                sample = generator.next()
            except StopIteration:
                generator = iter(data_loader)
                sample = generator.next()
            print(f"##### Epoch: {epoch}\t Iteration: {iteration}\t Sample: {sample}")
        print()
