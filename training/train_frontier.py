import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_root)

import torch
import numpy as np
import json
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score
from torchvision import transforms
from models.pnn_model import ProgressiveNeuralNetwork
from models.diffusion_model import DiffusionModel  # Import the generative model
from utils.complexity_assessment import assess_task_complexity
from utils.utils import set_device

# Loading and transforming the data
def load_task_data(task_id, json_dir='data/CIFAR-10_data_json/', train_ratio=0.8):
    file_path = os.path.join(json_dir, f'cifar10_task{task_id}.json')
    with open(file_path, 'r') as f:
        task_info = json.load(f)

    data, labels = [], []

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    for entry in task_info:
        img_path = entry["file_name"]
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        data.append(img.numpy())
        labels.append(entry['label'])

    data = np.array(data)
    labels = np.array(labels)

    indices = np.arange(len(data))
    np.random.shuffle(indices)
    split_idx = int(train_ratio * len(data))

    train_data = data[indices[:split_idx]]
    train_labels = labels[indices[:split_idx]]
    test_data = data[indices[split_idx:]]
    test_labels = labels[indices[split_idx:]]

    return train_data, train_labels, test_data, test_labels

# Evaluate per-class accuracy
def evaluate_per_class_accuracy(pred, labels):
    unique_labels = torch.unique(labels)
    per_class_accuracy = {}

    for label in unique_labels:
        correct = torch.sum((pred == label) & (labels == label)).item()
        total = torch.sum(labels == label).item()
        per_class_accuracy[label.item()] = correct / total if total > 0 else 0.0

    return per_class_accuracy

# Cosine similarity calculation
def cosine_similarity(tensor1, tensor2):
    tensor1 = tensor1.flatten()
    tensor2 = tensor2.flatten()
    return torch.nn.functional.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0)).item()

# Evaluate column performance
def evaluate_column_performance(pnn, task_id, accumulated_test_data, accumulated_test_labels):
    performances = []
    for i in range(len(pnn.columns)):
        output = pnn.forward_with_column(accumulated_test_data, task_id, i)
        pred = output.argmax(dim=1, keepdim=True).squeeze()
        accuracy = accuracy_score(accumulated_test_labels.cpu(), pred.cpu())
        performances.append(accuracy)
    return performances

# Remove redundant columns based on similarity and performance
def remove_redundant_columns(pnn, accumulated_test_data, accumulated_test_labels, similarity_threshold=0.9):
    performances = evaluate_column_performance(pnn, len(pnn.columns) - 1, accumulated_test_data, accumulated_test_labels)
    to_remove = set()
    
    for i in range(len(pnn.columns)):
        for j in range(i + 1, len(pnn.columns)):
            similarity = cosine_similarity(pnn.columns[i].state_dict()['0.weight'], pnn.columns[j].state_dict()['0.weight'])
            if similarity >= similarity_threshold:
                if performances[i] < performances[j]:
                    to_remove.add(i)
                else:
                    to_remove.add(j)

    if to_remove:
        print(f"Removing {len(to_remove)} redundant columns based on similarity...")
        pnn.columns = nn.ModuleList([col for i, col in enumerate(pnn.columns) if i not in to_remove])

def main():
    device = set_device()
    print(f"Using device: {device}")
    
    json_dir = '/Users/srirammandalika/Downloads/Minor/Codes/data/CIFAR-10_data_json/'
    
    input_dim = 32 * 32 * 3
    output_dim = 10
    tasks = 5
    pnn = ProgressiveNeuralNetwork(input_dim, output_dim, tasks).to(device)
    
    optimizer = torch.optim.Adam(pnn.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    test_accuracies = []
    all_test_data, all_test_labels = [], []

    # Define class names for CIFAR-10
    class_names = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"
    }

    for task_id in range(5):
        print(f"\n--- Task {task_id + 1} ---")

        if task_id > 0:
            pnn.add_column(task_id)
            pnn.columns[task_id].to(device)
        
        train_data, train_labels, test_data, test_labels = load_task_data(task_id + 1, json_dir=json_dir)
        
        train_data_flat = torch.tensor(train_data.reshape(len(train_data), -1), device=device)
        train_labels = torch.tensor(train_labels, device=device)
        test_data_flat = torch.tensor(test_data.reshape(len(test_data), -1), device=device)
        test_labels = torch.tensor(test_labels, device=device)

        all_test_data.append(test_data_flat)
        all_test_labels.append(test_labels)

        pnn.train()
        for epoch in range(30):
            optimizer.zero_grad()
            output = pnn(train_data_flat, task_id)
            loss = criterion(output, train_labels)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

        task_complexity = assess_task_complexity(train_data_flat.cpu().numpy(), train_labels.cpu().numpy())
        if task_complexity > 0.5:
            print("==> High task complexity detected. Upgrading model architecture (soft gating)...")
        else:
            print("==> Using hard gating.")
        
        if task_complexity < 0.3:
            print("==> Low task complexity detected. Removing redundant columns...")
            accumulated_test_data = torch.cat(all_test_data, dim=0)
            accumulated_test_labels = torch.cat(all_test_labels, dim=0)
            remove_redundant_columns(pnn, accumulated_test_data, accumulated_test_labels)

        pnn.eval()
        with torch.no_grad():
            accumulated_test_data = torch.cat(all_test_data, dim=0)
            accumulated_test_labels = torch.cat(all_test_labels, dim=0)
            output = pnn(accumulated_test_data, task_id)
            pred = output.argmax(dim=1, keepdim=True).squeeze()
            test_accuracy = accuracy_score(accumulated_test_labels.cpu(), pred.cpu())
            test_accuracies.append(test_accuracy)
            
            per_class_accuracy = evaluate_per_class_accuracy(pred, accumulated_test_labels)
            
            print(f"==> Test Accuracy in Task {task_id + 1}: {test_accuracy:.2f}")
            print(f"==> Class 1 Test Accuracy: {per_class_accuracy.get(1, 0.0):.2f}")
            print(f"==> Class 2 Test Accuracy: {per_class_accuracy.get(2, 0.0):.2f}")

            # Identify weak classes and print them with their accuracies
            weak_classes = {cls: acc for cls, acc in per_class_accuracy.items() if acc < 0.5}
            print(f"Weaker Classes: {weak_classes}")

            # Map weak classes to their actual class labels and print them
            mapped_weak_classes = {class_names[cls]: acc for cls, acc in weak_classes.items()}
            print(f"Mapped Weaker Classes: {mapped_weak_classes}")

            # Save the mapped weaker classes to a .pt file
            save_dir = 'Support Files'
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'original_{task_id + 1}.pt')
            torch.save(mapped_weak_classes, save_path)
            print(f"Saved mapped weaker classes to {save_path}")

    avg_test_accuracy = np.mean(test_accuracies)
    print(f"\n==> Average Test Accuracy across all tasks: {avg_test_accuracy:.2f}")

if __name__ == "__main__":
    main()
