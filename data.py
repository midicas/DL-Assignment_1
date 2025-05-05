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