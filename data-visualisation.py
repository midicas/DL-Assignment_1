import scipy.io
import matplotlib.pyplot as plt
import numpy as np


# RAW LASER MEASUREMENT TIME SERIES

# Load a .mat file,
mat = scipy.io.loadmat('Xtrain.mat')

# Extract the time series data
time_series_data = mat['Xtrain']

# Improved Printing 
print("Keys in the .mat file:", mat.keys())  # Show the keys to understand the structure
print("Shape of Xtrain:", time_series_data.shape) # Show the shape of the data
print("Data type of Xtrain:", time_series_data.dtype) # Show the data type

# Print the first few values:
print("\nFirst 10 values of Xtrain:\n", time_series_data[:10])

# Ensure time_series_data is 1D for plotting
time_series_data_1d = time_series_data.flatten()

plt.plot(time_series_data_1d)
plt.title('Raw Laser Measurement Time Series')
plt.xlabel('Time Step (t)')
plt.ylabel('Laser Measurement Valuelue')
plt.show()


#PREDICTIONS

"""
Time Series Visualization and Prediction Script
----------------------------------------------
This script:
1. Loads and visualizes the time series data
2. Loads a trained model
3. Makes recursive predictions for the next 200 time steps
4. Visualizes the predictions against the actual data
5. Computes evaluation metrics (MAE, MSE)

Author: Team ID
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import os
import argparse
from typing import Tuple, List, Optional

# Import your model classes
from models import MLP, RNN, Transformer, testModel


def load_data(file_path: str) -> np.ndarray:
    """
    Load time series data from a .mat file.
    
    Args:
        file_path: Path to the .mat file
        
    Returns:
        Time series data as a numpy array
    """
    try:
        mat = scipy.io.loadmat(file_path)
        # Extract the main data array - adjust key if necessary
        data = mat['Xtrain'].squeeze() if 'Xtrain' in mat else mat['Xtest'].squeeze()
        print(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def visualize_data(data: np.ndarray, title: str = "Time Series Data", 
                   save_path: Optional[str] = None) -> None:
    """
    Create and display a visualization of the time series data.
    
    Args:
        data: Time series data to visualize
        title: Plot title
        save_path: Path to save the figure (if None, will just display)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data, linewidth=1)
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def load_trained_model(model_type: str, window_size: int) -> torch.nn.Module:
    """
    Load a trained model from disk.
    
    Args:
        model_type: Type of model to load ('MLP', 'RNN', 'Transformer')
        window_size: Window size used during training
        
    Returns:
        Loaded PyTorch model
    """
    try:
        # Map string names to model classes
        model_map = {
            "MLP": MLP,
            "RNN": RNN,
            "Transformer": Transformer,
            "testModel": testModel
        }
        
        # Load the model file
        model_path = f"models/{model_type}/model_{model_type}.pt"
        model = torch.load(model_path)
        
        print(f"Model {model_type} loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        
        # If loading fails, create a new instance (fallback)
        if model_type == "RNN":
            return RNN()
        elif model_type == "Transformer":
            return Transformer(n=window_size, encode_pos=True)
        else:
            return model_map[model_type](n=window_size)


def make_recursive_predictions(model: torch.nn.Module, 
                              initial_sequence: np.ndarray,
                              scaler: StandardScaler,
                              num_predictions: int = 200) -> np.ndarray:
    """
    Make recursive predictions using the trained model.
    
    Args:
        model: Trained PyTorch model
        initial_sequence: Initial sequence to start predictions (length = window_size)
        scaler: Scaler used to normalize the data
        num_predictions: Number of future time steps to predict
        
    Returns:
        Array of predicted values
    """
    model.eval()  # Set model to evaluation mode
    
    # Convert the initial sequence to the right format
    current_sequence = initial_sequence.copy()
    
    # Scale the input sequence
    current_sequence_scaled = scaler.transform(current_sequence.reshape(-1, 1)).flatten()
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_predictions):
            # Convert to tensor
            sequence_tensor = torch.FloatTensor(current_sequence_scaled)
            
            # Add batch dimension if needed
            if len(sequence_tensor.shape) == 1:
                sequence_tensor = sequence_tensor.unsqueeze(0)
            
            # Get prediction
            output = model(sequence_tensor).item()
            
            # Store the prediction
            predictions.append(output)
            
            # Update the sequence (remove first element, add the prediction)
            current_sequence_scaled = np.append(current_sequence_scaled[1:], output)
    
    # Inverse transform the predictions
    # Create an array of shape (num_predictions, 1) for inverse_transform
    predictions_reshaped = np.array(predictions).reshape(-1, 1)
    predictions_rescaled = scaler.inverse_transform(predictions_reshaped).flatten()
    
    return predictions_rescaled


def evaluate_predictions(actual: np.ndarray, predicted: np.ndarray) -> Tuple[float, float]:
    """
    Calculate evaluation metrics for predictions.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        Tuple of (MAE, MSE)
    """
    # Calculate Mean Absolute Error
    mae = np.mean(np.abs(actual - predicted))
    
    # Calculate Mean Squared Error
    mse = np.mean((actual - predicted) ** 2)
    
    return mae, mse


def visualize_predictions(actual_data: np.ndarray, 
                         predictions: np.ndarray,
                         start_idx: int,
                         window_size: int,
                         title: str = "Recursive Predictions",
                         save_path: Optional[str] = None) -> None:
    """
    Visualize the predictions against the actual data.
    
    Args:
        actual_data: Full time series data
        predictions: Predicted values
        start_idx: Starting index in the actual data for predictions
        window_size: Size of input window used
        title: Plot title
        save_path: Path to save the figure
    """
    plt.figure(figsize=(14, 7))
    
    # Plot the window used for initial prediction
    window_indices = range(start_idx - window_size, start_idx)
    plt.plot(window_indices, actual_data[window_indices], 
             color='blue', label='Initial Window')
    
    # Plot the actual future values (if available)
    future_indices = range(start_idx, start_idx + len(predictions))
    if start_idx + len(predictions) <= len(actual_data):
        plt.plot(future_indices, actual_data[future_indices], 
                 color='green', label='Actual Future Values')
        
        # Calculate metrics if actual future values are available
        mae, mse = evaluate_predictions(
            actual_data[future_indices], predictions)
        metrics_text = f"MAE: {mae:.4f}, MSE: {mse:.4f}"
    else:
        metrics_text = "No actual future values for comparison"
    
    # Plot the predictions
    plt.plot(future_indices, predictions, 
             color='red', linestyle='--', linewidth=2, label='Predictions')
    
    # Add metrics as text on the plot
    plt.annotate(metrics_text, xy=(0.02, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction figure saved to {save_path}")
    
    plt.show()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Time Series Visualization and Prediction')
    parser.add_argument('--model', type=str, default='MLP', 
                        help='Model type (MLP, RNN, Transformer)')
    parser.add_argument('--window_size', type=int, default=50,
                        help='Input sequence length')
    parser.add_argument('--data_file', type=str, default='Xtrain.mat',
                        help='Data file path')
    parser.add_argument('--predict_steps', type=int, default=200,
                        help='Number of steps to predict')
    parser.add_argument('--start_idx', type=int, default=None,
                        help='Starting index for prediction (default: end of data - window_size)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Load and visualize data
    data = load_data(args.data_file)
    visualize_data(data, title=f"Time Series Data from {args.data_file}",
                   save_path=f"results/data_visualization.png")
    
    # Prepare data scaler
    scaler = StandardScaler()
    scaler.fit(data.reshape(-1, 1))
    
    # Load model
    model = load_trained_model(args.model, args.window_size)
    
    # Set start index if not provided
    if args.start_idx is None:
        args.start_idx = len(data) - args.window_size
    
    # Get initial sequence
    initial_sequence = data[args.start_idx - args.window_size:args.start_idx]
    
    # Make recursive predictions
    predictions = make_recursive_predictions(
        model, initial_sequence, scaler, args.predict_steps)
    
    # Visualize predictions
    visualize_predictions(
        data, predictions, args.start_idx, args.window_size,
        title=f"{args.model} - {args.predict_steps} Step Recursive Predictions",
        save_path=f"results/{args.model}_predictions.png")
    
    # Save predictions to file
    np.savetxt(f"results/{args.model}_predictions.csv", predictions, delimiter=',')
    print(f"Predictions saved to results/{args.model}_predictions.csv")
    
if __name__ == "__main__":
    main()