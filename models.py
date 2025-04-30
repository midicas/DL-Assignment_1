import torch
import torch.nn as nn

class testModel(torch.nn.Module):
    """
    Base model invoked for testing training functions
    """
    def __init__(self,n):
        super(testModel, self).__init__()
        self.linear = torch.nn.Linear(n,1)
        self.activation = torch.nn.ReLU()
    def forward(self,x):
        return self.activation(self.linear(x))
    
class MLP(torch.nn.Module):
    def __init__(self,n):
        super(MLP,self).__init__()
        self.linear = torch.nn.Linear(n,200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200,100)
        self.linear3 = torch.nn.Linear(100,1)
    
    def forward(self,x):
        out = self.linear(x)
        out = self.activation(out)
        out = self.linear2(out)
        out = self.linear3(out)

class RNN(torch.nn.Module):
    def __init__(self,n):
        super(RNN,self).__init__()
        self.linear = torch.nn.Linear(n,200)
        self.activation = torch.nn.ReLU()

