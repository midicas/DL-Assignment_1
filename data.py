from scipy.io import loadmat

import torch
from torch.utils.data import Dataset, DataLoader


class LaserDataset(Dataset):
    def __init__(self, data, seq_length=3):
        self.raw = torch.tensor(data, dtype=torch.uint8)
        #self.data_min = torch.min(self.raw)
        #self.data_max = torch.max(self.raw)
        #self.data = (self.raw - self.data_min) / (self.data_max - self.data_min)
        self.data = self.raw
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        return x, y