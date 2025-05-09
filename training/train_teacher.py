import sys
import os
import torch
import torch.nn as nn
import json
from torchvision.models import resnet18, ResNet18_Weights

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_root)

# Adding the utils directory to the path explicitly
utils_dir = os.path.join(project_root, '/Users/srirammandalika/Downloads/Minor/Codes/utils')
sys.path.append(utils_dir)

# Define paths (latent_dir will be passed as an argument or set dynamically)
latent_dir = '/Users/srirammandalika/Downloads/Minor/latent_vectors'  # Update as needed
output_json_path = '/Users/srirammandalika/Downloads/Minor/class_names.json'  # JSON save path

# Set device (using MPS for Apple Silicon or CPU if unavailable)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# CIFAR-10 class names (replace with your dataset class names if needed)
CIFAR10_CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Custom fully connected classification model (latent_dim and num_classes will be dynamic)
class LatentClassifier(nn.Module):
    def __init__(self, latent_dim=512, num_classes=10):  # CIFAR-10 has 10 classes
        super(LatentClassifier, self).__init__()
        self.fc = nn.Linear(latent_dim, num_classes)  # Simple fully connected layer for classification
    
    def forward(self, x):
        return self.fc(x)

# Modify ResNet18 to only use its fully connected layer for classification of latent vectors
def initialize_model(latent_dim, num_classes, pretrained=False):
    if pretrained:
        print("Using pretrained ResNet18's fully connected (fc) layer for classification.")
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(latent_dim, num_classes)  # Modify fc to take latent_dim as input
        model.forward = lambda x: model.fc(x)  # Modify forward to only use the fc layer
    else:
        print("Using custom latent classifier.")
        model = LatentClassifier(latent_dim=latent_dim, num_classes=num_classes).to(device)
    
    model = model.to(device)  # Move the model to the appropriate device (MPS or CPU)
    return model

def load_latent_vectors(task_id, latent_dir=latent_dir):
    file_path = os.path.join(latent_dir, f'latent_task_{task_id}.pt')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Latent vectors for Task {task_id} not found at {file_path}")
    
    latent_vectors = torch.load(file_path)
    print(f"Loaded latent vectors for Task {task_id}")
    return latent_vectors.to(device)

def predict_latent_vectors(model, latent_vectors):
    model = model.to(device)
    latent_vectors = latent_vectors.to(device)

    with torch.no_grad():
        outputs = model(latent_vectors)  # Pass latent vectors to the classifier
        _, predicted = torch.max(outputs, 1)  # Get the predicted class indices
    
    return predicted

def save_class_names_as_json(class_names_per_task, output_json_path):
    """Saves the predicted class names for each task in a JSON format."""
    with open(output_json_path, 'w') as f:
        json.dump(class_names_per_task, f, indent=4)
    print(f"Class names saved to {output_json_path}")

def main(latent_dir, num_classes=None, pretrained=False, output_json_path=None):
    # Use custom class names (e.g., CIFAR-10 class names)
    class_names = CIFAR10_CLASS_NAMES
    
    # Load one latent vector file to determine latent dimension dynamically
    first_task_id = 0
    first_latent_vectors = load_latent_vectors(first_task_id, latent_dir)
    
    latent_dim = first_latent_vectors.size(1)
    
    if num_classes is None:
        num_classes = 10  # CIFAR-10 has 10 classes

    model = initialize_model(latent_dim=latent_dim, num_classes=num_classes, pretrained=pretrained)
    
    model.eval()

    class_names_per_task = {}

    for task_id in range(5):  # Assuming we have 5 tasks
        latent_vectors = load_latent_vectors(task_id, latent_dir)
        pseudo_labels = predict_latent_vectors(model, latent_vectors)
        
        # Prepare class names per task
        unique_classes = torch.unique(pseudo_labels)
        task_class_names = {int(label): class_names[label] for label in unique_classes}
        class_names_per_task[f"Task {task_id + 1}"] = task_class_names

        # Output format
        print(f"Task {task_id + 1}:")
        for label, class_name in task_class_names.items():
            print(f"Class {label} => ({class_name})")
        print()  # New line for separation between tasks
    
    # Save class names as JSON
    if output_json_path:
        save_class_names_as_json(class_names_per_task, output_json_path)

if __name__ == "__main__":
    latent_dir = '/Users/srirammandalika/Downloads/Minor/latent_vectors'
    output_json_path = '/Users/srirammandalika/Downloads/Minor/class_names.json'
    main(latent_dir, pretrained=True, output_json_path=output_json_path)
