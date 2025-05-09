import torch
import sys
import os
import time  # Timing epochs

# Set recursion limit (temporary fix)
sys.setrecursionlimit(5000)

# Add project root directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_root)

import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
from torchvision import transforms
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from models.cae_model import CAE
from utils.utils import set_device

# -----------------------
# Paths
# -----------------------
save_tsne_dir = '/Users/srirammandalika/Downloads/Minor/data/tsne_plots/Loop_2/'
base_model_path = '/Users/srirammandalika/Downloads/Minor/main weights/Loop 1/cae_main_v1.pth'
synthetic_model_path = '/Users/srirammandalika/Downloads/Minor/synthetic_training/cae_synthetic_v1.pth'
updated_model_path = '/Users/srirammandalika/Downloads/Minor/main_weights/Loop_2/cae_main_v2.pth'

os.makedirs(save_tsne_dir, exist_ok=True)

# -----------------------
# Training Parameters
# -----------------------
num_epochs = 10
batch_size = 32
learning_rate = 0.01
tasks = 5
clusters_per_task = 2  # 2 clusters per task
total_clusters = tasks * clusters_per_task

# -----------------------
# Load Known Classes
# -----------------------
def load_known_classes(synthetic_path):
    """
    Load known class names from synthetic model weights.
    """
    checkpoint = torch.load(synthetic_path, map_location='cpu', weights_only=True)  # Safe loading

    # Extract class names if present
    if 'class_names' in checkpoint and isinstance(checkpoint['class_names'], list):
        return checkpoint['class_names']
    else:
        print("Warning: 'class_names' key missing in checkpoint. Returning empty list.")
        return []


# -----------------------
# Load Unlabelled Task Data
# -----------------------
def load_unlabelled_task_data(task_id, json_dir='/Users/srirammandalika/Downloads/Minor/data/CIFAR-10_data_json/', train_ratio=0.8):
    file_path = os.path.join(json_dir, f'cifar10_task{task_id}.json')

    with open(file_path, 'r') as f:
        task_info = json.load(f)

    data = []
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    for entry in task_info:
        img_path = entry.get("file_name", "")
        img = transform(Image.open(img_path).convert('RGB'))
        data.append(img.numpy())

    data = np.array(data)
    split_idx = int(train_ratio * len(data))
    return data[:split_idx], data[split_idx:]


# -----------------------
# TSNE Plotting
# -----------------------
def map_cluster_labels(cluster_labels, known_class_names):
    """
    Map cluster IDs to known class names or default cluster labels.
    """
    mapped_labels = []
    for cluster_id in cluster_labels:
        if cluster_id < len(known_class_names):
            mapped_labels.append(known_class_names[cluster_id])  # Use known names
        else:
            mapped_labels.append(f"Cluster_{cluster_id}")  # Default unknown label
    return mapped_labels


def plot_tsne(latent_vectors, cluster_labels, save_path, known_class_names, title_suffix=""):
    """
    TSNE visualization with convex hulls, density highlights, and class labeling.
    """
    tsne = TSNE(n_components=2, perplexity=min(30, len(latent_vectors) - 1), random_state=99)
    reduced_vectors = tsne.fit_transform(latent_vectors)

    # Map cluster labels dynamically
    label_names = map_cluster_labels(cluster_labels, known_class_names)

    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(
        reduced_vectors[:, 0], reduced_vectors[:, 1], c=cluster_labels, cmap='tab10', alpha=0.6, s=40, edgecolors='k'
    )
    plt.colorbar(scatter, label="Cluster")
    plt.title(f"TSNE of Latent Space {title_suffix}")
    plt.xlabel("TSNE Dimension 1")
    plt.ylabel("TSNE Dimension 2")
    plt.grid(True)

    # Highlight clusters
    for cluster_id in np.unique(cluster_labels):
        cluster_points = reduced_vectors[np.array(cluster_labels) == cluster_id]

        # Density calculation
        kde = gaussian_kde(cluster_points.T)
        density = kde(cluster_points.T)
        densest_point = cluster_points[np.argmax(density)]
        plt.scatter(densest_point[0], densest_point[1], c='red', s=150, marker='X')  # Highlight densest point

        # Draw convex hull if enough points
        if len(cluster_points) >= 3:
            hull = ConvexHull(cluster_points)
            for simplex in hull.simplices:
                plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 'k-', linewidth=2)

        # Label clusters with mapped names
        plt.text(
            cluster_points[:, 0].mean(), cluster_points[:, 1].mean(),
            label_names[cluster_id], fontsize=12, weight='bold'
        )

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"TSNE plot saved to {save_path}")


# -----------------------
# Model Initialization
# -----------------------
def initialize_model(device, base_path, synthetic_path):
    cae = CAE().to(device)

    # Load base model weights safely
    cae.load_state_dict(torch.load(base_path, map_location=device, weights_only=True))
    print(f"Loaded base CAE weights from {base_path}")

    # Integrate synthetic knowledge safely
    synthetic_weights = torch.load(synthetic_path, map_location=device, weights_only=True)
    cae.load_state_dict(synthetic_weights, strict=False)
    print(f"Integrated synthetic CAE weights from {synthetic_path}")

    optimizer = torch.optim.AdamW(cae.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    return cae, optimizer, scheduler


# -----------------------
# Main Function
# -----------------------
def main():
    device = set_device()

    # Load known classes
    known_class_names = load_known_classes(synthetic_model_path)
    print(f"Known Class Names: {known_class_names}")

    if not known_class_names:
        print("No class names found. Check the synthetic model checkpoint.")
        return

    cae, optimizer, scheduler = initialize_model(device, base_model_path, synthetic_model_path)
    criterion = torch.nn.MSELoss()

    for task_id in range(1, tasks + 1):
        train_data, _ = load_unlabelled_task_data(task_id)
        loader = DataLoader(TensorDataset(torch.tensor(train_data, dtype=torch.float32).to(device)), batch_size=batch_size, shuffle=True)

        cae.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for x in loader:
                optimizer.zero_grad()
                recon = cae(x[0])
                loss = criterion(recon, x[0])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()

        latents = np.vstack([cae.encoder(x[0].to(device)).view(x[0].size(0), -1).detach().cpu().numpy() for x in loader])
        clusters = GaussianMixture(n_components=clusters_per_task).fit_predict(latents)

        plot_tsne(latents, clusters, os.path.join(save_tsne_dir, f"task_{task_id}_latent_space.png"), known_class_names, f"Task {task_id}")

if __name__ == "__main__":
    main()
