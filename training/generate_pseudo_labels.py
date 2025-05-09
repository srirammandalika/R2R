import sys
import os
import torch
from torch.utils.data import DataLoader

# Add the project directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_root)

from models.teacher_model import TeacherModel
from data.cifar10 import get_cifar10_data
from utils.utils import set_device, save_pseudo_labels

def generate_pseudo_labels(model, data_loader, device):
    model.eval()
    pseudo_labels = []

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            pseudo_labels.extend(predicted.cpu().numpy())
    
    return pseudo_labels

def main():
    # Set device
    device = set_device()
    print(f"Using device: {device}")

    # Define hyperparameters
    batch_size = 64

    # Get CIFAR-10 data
    train_loader, _ = get_cifar10_data(batch_size, data_dir='/Users/srirammandalika/Downloads/Minor/CIFAR-10 data/cifar10/')

    # Initialize the teacher model
    model = TeacherModel().to(device)
    model.load_state_dict(torch.load('./models/Pretrained_teacher_model.pth', map_location=device))

    # Generate pseudo labels
    pseudo_labels = generate_pseudo_labels(model, train_loader, device)

    # Ensure the directory exists
    pseudo_labels_dir = './data/pseudo_labels/'
    os.makedirs(pseudo_labels_dir, exist_ok=True)

    # Save pseudo labels
    save_pseudo_labels(pseudo_labels, os.path.join(pseudo_labels_dir, 'pseudo_labels.npy'))

if __name__ == "__main__":
    main()
