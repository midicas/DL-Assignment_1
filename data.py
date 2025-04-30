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

# Print the first batch
for x_batch, y_batch in loader:
    print("Input sequence (x):", x_batch)
    print("Target (y):", y_batch)
    break
