from scipy.io import loadmat

raw = loadmat("Xtrain.mat")["Xtrain"].squeeze()

import torch
from torch.utils.data import Dataset, DataLoader


class LaserDataset(Dataset):
    def __init__(self, data, seq_length=3):
        self.data = torch.tensor(data, dtype=torch.uint8)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        return x, y


dataset = LaserDataset(raw, seq_length=5)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Example usage
from models import testModel

model = testModel(n=5)

for x_batch, y_batch in loader:
    output = model(x_batch.float())
    print("Input batch (x):")
    print(x_batch)
    print("Model output:")
    print(output)
    print("Target (y):")
    print(y_batch)
    break 