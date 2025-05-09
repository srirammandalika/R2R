import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from cae_model import CAE  # Your CAE model file


def compute_loss_and_accuracy(model, data_loader, device, threshold=0.01):
    """
    Computes average MSE loss over the given data_loader.
    Also computes a 'reconstruction accuracy': fraction of images
    whose MSE < threshold.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0  # how many samples have MSE < threshold

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            outputs = model(images)

            # MSE for the batch (scalar)
            batch_loss = model.loss_function(outputs, images)

            batch_size = images.size(0)
            total_loss += batch_loss.item() * batch_size
            total_samples += batch_size

            # For 'accuracy': compute per-sample MSE, check if it's < threshold
            #   shape: (batch_size, C, H, W)
            # Flatten each image to (batch_size, C*H*W) and do MSE along dim=1
            mse_per_sample = ((outputs - images) ** 2).view(batch_size, -1).mean(dim=1)
            correct += (mse_per_sample < threshold).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy


def train_cae_cifar10(
    data_dir,
    batch_size=64,
    epochs=10,
    lr=1e-3,
    device="mps"
):
    """
    Train the CAE model on the entire CIFAR-10 dataset,
    splitting 90% for training and 10% for testing.
    Also prints train/test loss and train/test accuracy each epoch.
    """
    # 1) Define transforms
    transform = transforms.Compose([
        transforms.ToTensor()
        # For autoencoders, it's often okay to skip or do minimal normalization
    ])

    # 2) Load official CIFAR-10 train + test
    cifar_train = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=transform,
        download=False  # set True once if needed
    )
    cifar_test = datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform=transform,
        download=False
    )

    # Combine them into one 60K dataset
    full_data = ConcatDataset([cifar_train, cifar_test])
    full_size = len(full_data)  # should be 60,000

    # 3) Manually split 90%/10%
    train_size = int(0.9 * full_size)  # 54,000
    test_size = full_size - train_size  # 6,000
    train_dataset, test_dataset = random_split(
        full_data,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # seed for reproducibility
    )

    # 4) Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # 5) Initialize Model & Optimizer
    model = CAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 6) Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_train = 0

        for images, _ in train_loader:
            images = images.to(device)
            outputs = model(images)
            loss = model.loss_function(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_ = images.size(0)
            running_loss += loss.item() * batch_size_
            total_train += batch_size_

        train_loss_epoch = running_loss / total_train

        # Evaluate on train set (optional, for seeing train metrics) and test set
        train_loss_eval, train_acc_eval = compute_loss_and_accuracy(model, train_loader, device)
        test_loss_eval, test_acc_eval = compute_loss_and_accuracy(model, test_loader, device)

        # Print logs
        print(f"Epoch [{epoch}/{epochs}]")
        print(f"  Train Loss (batch avg): {train_loss_epoch:.6f}")
        print(f"  Train Loss (eval pass): {train_loss_eval:.6f},  Train Acc: {train_acc_eval:.4f}")
        print(f"  Test  Loss (eval pass):  {test_loss_eval:.6f},   Test  Acc: {test_acc_eval:.4f}")

    # 7) Save the model
    save_path = os.path.join(data_dir, "cae_cifar10.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}.")


if __name__ == "__main__":
    # Path to your CIFAR-10 data directory
    cifar10_data_dir = "/Users/srirammandalika/Downloads/Minor/CIFAR-10 data/cifar10/"

    # Choose device (Apple Silicon = "mps", otherwise "cuda" or "cpu")
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    train_cae_cifar10(
        data_dir=cifar10_data_dir,
        batch_size=16,
        epochs=10,
        lr=1e-3,
        device=device
    )
