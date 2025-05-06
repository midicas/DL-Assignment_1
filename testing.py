from models import *
from training import *

if __name__ == "__main__":
    parameter_space_MLP = {"seq_length": [50],#list(np.arange(1,51,5)),
                       "Optimizer": [torch.optim.Adam,torch.optim.AdamW],
                       "Batch_size": [1,8,16,32],
                       "Epochs": [1,25,50,100],
                       "Learning_rate":[0.1,0.01,0.001]}
    grid_search(MLP,parameter_space_MLP)
    
    parameter_space_RNN = {"seq_length": [50],#list(np.arange(1,51,5)),
                       "Optimizer": [torch.optim.Adam,torch.optim.AdamW],
                       "Batch_size": [1,8,16,32],
                       "Epochs": [1,25,50,100],
                       "Learning_rate":[0.1,0.01,0.001],
                       "hidden_size": [1,16,32,64],
                       "num_of_layers": [1,2,4,8]}
    grid_search(RNN,parameter_space_RNN)

    parameter_space_TF = {"seq_length": [50],#list(np.arange(1,51,5)),
                       "Optimizer": [torch.optim.Adam,torch.optim.AdamW],
                       "Batch_size": [1,8,16,32],
                       "Epochs": [1,25,50,100],
                       "Learning_rate":[0.1,0.01,0.001],
                       "pos": [True,False],
                       "dimensionality": [1,16,32,64],
                       "transformer_blocks": [1,2,4,8]}
    grid_search(Transformer,parameter_space_TF)