from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import argparse
import os

from models import testModel, MLP, Transformer, RNN
from data import LaserDataset


def train_model(epochs: int = 1, train_loader: DataLoader = None, validation_loader: DataLoader = None, model: torch.nn.Module = None, loss_function = None, optimizer: torch.optim.Optimizer = None) -> None:
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

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for x_batch, y_batch in train_loader:
                output = model(x_batch.float())

                loss = loss_function(output, y_batch.float())
                val_losses.append(loss.item())
            
        avg_val_loss = np.mean(val_losses)

        print(f"Epoch {i} | Training Loss: {avg_train_loss:.3f} | Validation Loss: {avg_val_loss:.3f}")


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

    train_model(epochs, train_loader, validation_loader, model_, loss_function, optimizer)
    save_model(model)


def save_model(model: torch.nn.Module) -> None:
    os.makedirs("models", exist_ok=True)
    torch.save(model, f"models/model_{model.__name__}.pt")


def load_model(model_name: str) -> torch.nn.Module:
    return torch.load(f"models/model_{model_name}.pt", weights_only=False)


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
