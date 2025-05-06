import torch
import torch.nn as nn
import math

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
        self.linear3 = torch.nn.Linear(100,50)
        self.output = torch.nn.Linear(50,1)
    
    def forward(self,x):
        out = self.linear(x)
        out = self.activation(out)
        out = self.linear2(out)
        out = self.activation(out)
        out = self.linear3(out)
        out = self.activation(out)
        out = self.output(out)
        return out

class RNN(torch.nn.Module):
    def __init__(self, hidden_size = 64, num_layers=4):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(1, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.rnn(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)

class Transformer(nn.Module):
    def __init__(self, n,encode_pos, model_dimensionality=64, num_of_transformer_blocks=2):
        super(Transformer, self).__init__()
        #Learns embeddings of simple scalar (Might be overkill)
        self.embedding = nn.Linear(1, model_dimensionality)
        #Creates positional encodings
        self.encode_pos = encode_pos
        self.pos_encoder = PositionalEncoding(model_dimensionality, max_len=n)
        
        #Create transformer layer then stack layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dimensionality, nhead=1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_of_transformer_blocks)
        
        #output layer can be replaced by a different model
        self.output = nn.Linear(model_dimensionality, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        
        if self.encode_pos:
            x = self.pos_encoder(x)
        
        x = x.permute(1, 0, 2)
        out = self.transformer(x)
        out = out[-1]
        return self.output(out)
    
class PositionalEncoding(nn.Module):
    """
    Layer that applies sinusoidal position encoding.
    """
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1).float()  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return x
