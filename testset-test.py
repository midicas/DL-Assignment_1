import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from scipy.io import loadmat
from torch.utils.data import DataLoader
from data import LaserDataset

# Load the models
# Run train-optimal.py if you don't have them
models = [{
    "modelName": "MLP",
    "model": torch.load("./models/MLP/model_MLP.pt", weights_only=False),
    "windowSize": 16,
    "predictions": [],
    "targets": []
}, {
    "modelName": "RNN",
    "model": torch.load("./models/RNN/model_RNN.pt", weights_only=False),
    "windowSize": 8,
    "predictions": [],
    "targets": []
}, {
    "modelName": "Transformer",
    "model": torch.load("./models/Transformer/model_Transformer.pt", weights_only=False),
    "windowSize": 27,
    "predictions": [],
    "targets": []
}]

# Load the test dataset
raw = loadmat("Xtest.mat")["Xtest"].squeeze()

# Use CUDA if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run the predictions
for model_dict in models:
    dataset = LaserDataset(raw, seq_length=model_dict['windowSize'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = model_dict["model"]
    model.eval()
    model.to(device)

    predictions = []

    with torch.no_grad():
        for i, inputs in enumerate(dataloader):
            if isinstance(inputs, (list, tuple)):
                inputs = inputs[0]

            inputs = inputs.float().to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu())

            # Ground truth value
            target_index = i + model_dict['windowSize']
            if target_index < len(raw):
                model_dict["targets"].append(torch.tensor(raw[target_index]))

    # Save the predictions to plot later
    model_dict["predictions"] = predictions

#########################
# Quantitative Analysis #
#########################
# Compute the quality of the predictions
print(f"{'Model':<12} {'MSE':>12} {'MAE':>12}")
print("-" * 38)
for model_dict in models:
    predictions = torch.cat(model_dict["predictions"]).squeeze()
    targets = torch.stack(model_dict["targets"]).squeeze()

    mse = F.mse_loss(predictions, targets)
    mae = F.l1_loss(predictions, targets)

    print(f"{model_dict['modelName']:<12} {mse.item():>12.4f} {mae.item():>12.4f}")

########################
# Qualitative Analysis #
########################
os.makedirs("plots", exist_ok=True)

for model_dict in models:
    model_name = model_dict["modelName"]
    predictions = torch.cat(model_dict["predictions"]).squeeze().numpy()
    targets = torch.stack(model_dict["targets"]).squeeze().numpy()

    # Prediction vs Ground Truth
    plt.figure(figsize=(10, 4))
    plt.plot(targets, label="Ground Truth", linewidth=1)
    plt.plot(predictions, label="Prediction", linewidth=1)
    plt.title(f"{model_name} Prediction vs Ground Truth")
    plt.xlabel("Time Step")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_pred_vs_truth.png")
    plt.close()

    # Error plot
    error = abs(predictions - targets)
    plt.figure(figsize=(10, 3))
    plt.plot(error)
    plt.title(f"{model_name} Absolute Error Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Absolute Error")
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_error.png")
    plt.close()
