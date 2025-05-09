import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_root)

import sys
import os
import torch
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from models.vae_model import VAE, train_vae, extract_latent_vectors
from utils.utils import set_device
from PIL import Image

def load_unlabelled_task_data(task_id, json_dir='/Users/srirammandalika/Downloads/Minor/data/CIFAR-10_data_json/', train_ratio=0.8):
    """
    Load and preprocess unlabelled task data from JSON files.
    """
    file_path = os.path.join(json_dir, f'cifar10_task{task_id}.json')
    try:
        with open(file_path, 'r') as f:
            task_info = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for Task {task_id}: {e}")
        print(f"Skipping Task {task_id} due to unrecoverable JSON issues.")
        return [], []

    data = []

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    for entry in task_info:
        try:
            img_path = entry.get("file_name", "")
            if not img_path:
                print(f"Skipping invalid entry: {entry}")
                continue

            img = transform(Image.open(img_path).convert('RGB'))
            data.append(img.numpy())
        except Exception as e:
            print(f"Error processing entry {entry}: {e}")
            continue

    data = np.array(data)

    # Shuffle and split data
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    split_idx = int(train_ratio * len(data))

    train_data = data[indices[:split_idx]]
    test_data = data[indices[split_idx:]]

    return train_data, test_data

def apply_gmm(latent_vectors, n_components=10):
    """
    Apply Gaussian Mixture Model clustering to latent vectors.

    Args:
        latent_vectors (np.array): The latent vectors from the VAE.
        n_components (int): Number of Gaussian components/clusters.

    Returns:
        cluster_labels (np.array): The cluster labels for each sample.
        gmm_model (GaussianMixture): The fitted GMM model.
    """
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    cluster_labels = gmm.fit_predict(latent_vectors)
    return cluster_labels, gmm

def plot_tsne(latent_vectors, cluster_labels, task_id=None, save_dir='/Users/srirammandalika/Downloads/Minor/data/tsne_plots/', title_suffix="", n_components=2):
    """Generate and save TSNE plots for the given latent vectors and cluster labels."""
    tsne = TSNE(n_components=n_components, random_state=42)
    reduced_vectors = tsne.fit_transform(latent_vectors)

    if n_components == 2:
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=cluster_labels, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter, label="Cluster")
        plt.title(f"TSNE of Latent Space {title_suffix}")
        plt.xlabel("TSNE Dimension 1")
        plt.ylabel("TSNE Dimension 2")
        plt.grid(True)
    elif n_components == 3:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], reduced_vectors[:, 2],
                             c=cluster_labels, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter, label="Cluster")
        ax.set_title(f"TSNE of Latent Space {title_suffix}")
        ax.set_xlabel("TSNE Dimension 1")
        ax.set_ylabel("TSNE Dimension 2")
        ax.set_zlabel("TSNE Dimension 3")
        plt.grid(True)

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    suffix = "3D" if n_components == 3 else "2D"
    if task_id is not None:
        save_path = os.path.join(save_dir, f"tsne_task_{task_id}_{suffix}.png")
    else:
        save_path = os.path.join(save_dir, f"tsne_all_tasks_{suffix}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"TSNE {suffix} plot saved to {save_path}")

def main():
    device = set_device()
    print(f"Using device: {device}")
    
    json_dir = '/Users/srirammandalika/Downloads/Minor/data/CIFAR-10_data_json/'

    # Containers for all-task latent vectors and cluster labels
    all_latent_vectors = []
    all_cluster_labels = []
    task_offsets = []  # To adjust cluster labels across tasks

    for task_id in range(1, 6):
        print(f"\n--- Task {task_id} ---")

        # Load task-specific data
        train_data, test_data = load_unlabelled_task_data(task_id, json_dir=json_dir)

        # Skip task if data is empty
        if len(train_data) == 0 or len(test_data) == 0:
            print(f"Skipping Task {task_id} due to empty data.")
            continue

        # Train and cluster using VAE
        input_dim = 32 * 32 * 3  # CIFAR-10 flattened input size
        latent_dim = 64  # Adjustable latent space dimension

        vae = VAE(input_dim=input_dim, latent_dim=latent_dim)
        optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

        train_tensor = torch.tensor(train_data.reshape(len(train_data), -1), dtype=torch.float32)
        train_loader = DataLoader(TensorDataset(train_tensor), batch_size=32, shuffle=True)

        # Train VAE
        train_vae(vae, train_loader, optimizer, device, epochs=10)

        # Extract latent vectors
        latent_vectors = extract_latent_vectors(vae, train_loader, device)

        # Apply GMM clustering
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=2, random_state=42)  # Expecting 2 clusters per task
        gmm.fit(latent_vectors)
        cluster_labels = gmm.predict(latent_vectors)

        # Save clustering results
        cluster_path = f'/Users/srirammandalika/Downloads/Minor/latents/cluster_labels_task_{task_id}.pt'
        torch.save(torch.tensor(cluster_labels), cluster_path)
        print(f"Saved cluster labels to {cluster_path}")

        # Save TSNE plots (2D and 3D) for the current task
        plot_tsne(latent_vectors, cluster_labels, task_id=task_id, save_dir='/Users/srirammandalika/Downloads/Minor/data/tsne_plots/', title_suffix=f"Task {task_id}", n_components=2)
        plot_tsne(latent_vectors, cluster_labels, task_id=task_id, save_dir='/Users/srirammandalika/Downloads/Minor/data/tsne_plots/', title_suffix=f"Task {task_id}", n_components=3)

        # Accumulate data for all-tasks TSNE
        task_offsets.append(len(all_latent_vectors))
        all_latent_vectors.extend(latent_vectors)
        all_cluster_labels.extend([label + sum(task_offsets) for label in cluster_labels])

    # Plot TSNE for all tasks (2D and 3D) if data is available
    if all_latent_vectors and all_cluster_labels:
        plot_tsne(np.array(all_latent_vectors), np.array(all_cluster_labels), save_dir='/Users/srirammandalika/Downloads/Minor/data/tsne_plots/', title_suffix="All Tasks", n_components=2)
        plot_tsne(np.array(all_latent_vectors), np.array(all_cluster_labels), save_dir='/Users/srirammandalika/Downloads/Minor/data/tsne_plots/', title_suffix="All Tasks", n_components=3)


if __name__ == "__main__":
    main()
