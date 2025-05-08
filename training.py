from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import argparse
import os
import matplotlib.pyplot as plt
from models import testModel, MLP, Transformer, RNN
from data import LaserDataset
from itertools import product


def train_model(epochs: int = 1, train_loader: DataLoader = None, validation_loader: DataLoader = None, model: torch.nn.Module = None, loss_function = None, optimizer: torch.optim.Optimizer = None) -> None:
    avg_train_losses = []
    avg_val_losses = []

    for i in range(epochs):
        model.train()
        train_losses = []

        for x_batch, y_batch in train_loader:
            output = model(x_batch.float())

            loss = loss_function(output, y_batch.float())
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = np.mean(train_losses)
        avg_train_losses.append(avg_train_loss)
        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for x_batch, y_batch in validation_loader:
                output = model(x_batch.float())

                loss = loss_function(output, y_batch.float())
                val_losses.append(loss.item())
            
        avg_val_loss = np.mean(val_losses)
        avg_val_losses.append(avg_val_loss)
        print(f"Epoch {i} | Training Loss: {avg_train_loss:.10f} | Validation Loss: {avg_val_loss:.10f}")

    return avg_train_losses,avg_val_losses
    

def train(model: torch.nn.Module=MLP, window_size: int=15, learning_rate: float=0.00003, batch_size: int=32, epochs: int=100):
    """Loads the training data, model, loss function, and optimizer, and trains the model. After training, the model is saved to the 'models' folder.

    Keyword arguments:
        model: class name of the model to be trained
        window_size: length of input sequence
        learning_rate: learning rate passed to the optimizer
        batch_size: number of training samples per batch
        epochs: number of training iterations
    """

    print("Using model " + model.__name__)
    print("\t learning rate: " + str(learning_rate))
    print("\t number of epochs: " + str(epochs))
    print("\t batch size: " + str(batch_size))

    raw = loadmat("Xtrain.mat")["Xtrain"].squeeze()
    dataset = LaserDataset(raw, seq_length=window_size)

    train_size = int(0.8*len(dataset))

    validation_size = len(dataset) - train_size

    # split the training dataset
    train_data, validation_data = random_split(dataset, [train_size, validation_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

    model_ = model() if model == RNN else model(n=window_size, encode_pos=True) if model == Transformer else model(n=window_size)
    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.NAdam(model_.parameters(), lr=learning_rate)

    train_losses,val_losses = train_model(epochs, train_loader, validation_loader, model_, loss_function, optimizer)
    
    os.makedirs(f"models/{model.__name__}", exist_ok=True)
    create_graph(train_losses,val_losses,model.__name__)
    
    save_model(model)

def grid_search(model : torch.nn.Module,parameter_space : dict):
    raw = loadmat("Xtrain.mat")["Xtrain"].squeeze()
    
    best_score = float('inf')
    best_parameters = None

    results = []
    for seq_length,optimizer_fn,batch_size,epochs,learning_rate,*model_params in product(*parameter_space.values()):
        dataset = LaserDataset(raw, seq_length=seq_length)

        train_size = int(0.8*len(dataset))

        validation_size = len(dataset) - train_size

        # split the training dataset
        train_data, validation_data = random_split(dataset, [train_size, validation_size])

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

        model_ = testModel(seq_length)
        if model.__name__ == "MLP":
            model_ = model(seq_length)
            print(f"Training {model.__name__} model: Window size:{seq_length}, Optimizer:{optimizer_fn.__name__}, batch size: {batch_size}, epochs: {epochs}, learning rate:{learning_rate}")
        elif model.__name__ == "RNN":
            hidden_size,num_layers = model_params
            model_ = model(hidden_size,num_layers)
            print(f"Training {model.__name__} model: Window size:{seq_length}, Optimizer:{optimizer_fn.__name__}, batch size: {batch_size}, epochs: {epochs}, learning rate:{learning_rate},hidden size: {hidden_size},num layers:{num_layers}")
        elif model.__name__ == "Transformer":
            encode_pos, model_dimensionality, num_of_transformer_blocks = model_params
            model_ = model(seq_length,encode_pos,model_dimensionality,num_of_transformer_blocks)
            print(f"Training {model.__name__} model: Window size:{seq_length}, Optimizer:{optimizer_fn.__name__}, batch size: {batch_size}, epochs: {epochs}, learning rate:{learning_rate},position encoding: {encode_pos},num layers:{num_of_transformer_blocks}, model dimensionality: {model_dimensionality}")
        
        optimizer = optimizer_fn(model_.parameters(),lr = learning_rate)
        loss_function = torch.nn.L1Loss()
        _,val_losses = train_model(epochs,train_loader,validation_loader,model_,loss_function,optimizer)
        results.append(val_losses[-1])
        if val_losses[-1] < best_score:
            best_score = val_losses[-1]
            best_parameters = [seq_length,optimizer_fn,batch_size,epochs,learning_rate,*model_params]
    file = open(f'models/{model.__name__}/best_parameters.txt','w')
    for parameter in best_parameters:
        file.write(str(parameter) + "\n")
    file.close()
    print(f"Best score: {best_score}, Best parameters: {best_parameters}")
    
    return results

def create_graph(train_losses: list, val_losses : list, model_name : str) -> None:
    plt.plot(train_losses,label = "Training")
    plt.plot(val_losses,label = "Validation")
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(model_name + " Results")
    plt.savefig(f"models/{model_name}/{model_name}_results", dpi=300)
    plt.close()


def save_model(model: torch.nn.Module) -> None:
    
    torch.save(model, f"models/{model.__name__}/model_{model.__name__}.pt")


def load_model(model_name: str) -> torch.nn.Module:
    return torch.load(f"models/{model_name}/model_{model_name}.pt", weights_only=False)


def get_model(model_name: str) -> torch.nn.Module:
    models = [testModel, MLP, RNN, Transformer]
    for m in models:
        if model_name == m.__name__:
            return m
    
    return MLP

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', help='Model', type=str, default='MLP')
    parser.add_argument('-l', help='Learning rate', type=float, default=0.00003)
    parser.add_argument('-w', help='Window size', type=int, default=15)
    parser.add_argument('-e', help='Number of epochs', type=int, default=100)
    parser.add_argument('-b', help='Batch size', type=int, default=32)
    args = parser.parse_args()

    model = get_model(args.m)

    train(model, args.w, args.l, args.b, args.e)
