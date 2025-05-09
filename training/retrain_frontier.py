# training/retrain_frontier.py

import torch
from torch.utils.data import TensorDataset, DataLoader
from models.pnn_model import ProgressiveNeuralNetwork
from utils.utils import set_device

def retrain_with_synthetic_data(model, train_data, train_labels, synthetic_data, synthetic_labels, task_id):
    """
    Retrain the model with the combined original and synthetic data.

    Args:
        model (ProgressiveNeuralNetwork): The Frontier model.
        train_data (torch.Tensor): Original training data.
        train_labels (torch.Tensor): Original training labels.
        synthetic_data (torch.Tensor): Generated synthetic data.
        synthetic_labels (torch.Tensor): Generated synthetic labels.
        task_id (int): The current task ID.
    """
    device = set_device()
    model = model.to(device)

    # Combine original and synthetic data
    combined_train_data = torch.cat((train_data, synthetic_data), dim=0)
    combined_train_labels = torch.cat((train_labels, synthetic_labels), dim=0)

    # Create DataLoader
    dataset = TensorDataset(combined_train_data, combined_train_labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(30):  # Adjust the number of epochs as needed
        for batch_data, batch_labels in dataloader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            output = model(batch_data, task_id)
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()
        print(f"Retrain Epoch {epoch + 1}, Retrain Loss: {loss.item()}")

if __name__ == "__main__":
    # Example usage
    task_id = 1  # Replace with the actual task_id you are retraining for

    # Load your trained Frontier model
    model = ProgressiveNeuralNetwork(input_dim=32*32*3, output_dim=10, tasks=5)

    # Load original data
    train_data = torch.load("train_data.pt")
    train_labels = torch.load("train_labels.pt")

    # Load synthetic data
    synthetic_data = torch.load("synthetic_data.pt")
    synthetic_labels = torch.load("synthetic_labels.pt")

    # Retrain model with synthetic data
    retrain_with_synthetic_data(model, train_data, train_labels, synthetic_data, synthetic_labels, task_id)
