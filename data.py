from scipy.io import loadmat

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

def createData(fname,seq_length):
    raw = loadmat("Xtrain.mat")["Xtrain"].squeeze()
    
    dataset = LaserDataset(raw, seq_length=5)
    
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    return loader
# Example usage

if __name__ == "__main__":
    from models import TransformerWithPE

    model = TransformerWithPE(n=5)
    
    loader = createData("Xtrain.mat",5)

    for x_batch, y_batch in loader:
        output = model(x_batch.float())
        print("Input batch (x):")
        print(x_batch)
        print("Model output:")
        print(output)
        print("Target (y):")
        print(y_batch)
        break 