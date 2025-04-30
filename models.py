import torch
import torch.nn as nn

class MLP(torch.nn.Module):
    def __init__(self,n):
        super(MLP,self).__init__()
        self.linear = torch.nn.Linear(n,200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200,1)
    def forward(x):
        out = linear(x)
        out = activation
