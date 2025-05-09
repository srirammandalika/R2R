import sys
import os
import json
import torch
import numpy as np
import scipy
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from scipy.spatial import distance, ConvexHull
import matplotlib.pyplot as plt
from PIL import Image

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_root)

from models.cae_model import CAE  # Ensure the correct path

# ---------------------------
# Paths
# ---------------------------
data_dir = '/Users/srirammandalika/Downloads/Minor/synthetic_training/GR Samples/'
model_path = '/Users/srirammandalika/Downloads/Minor/main weights/Loop 1/cae_main_v1.pth'
output_dir = '/Users/srirammandalika/Downloads/Minor/uncertainty_analysis/clusters/'
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# Device Setup
# ---------------------------
def set_device():
    """Set the device to MPS if available, otherwise fallback to CPU."""
    if torch.backends.mps.is_available():
        print("[INFO] Using MPS device.")
        return torch.device("mps")
    else:
        print("[INFO] Using CPU.")
        return torch.device("cpu")

# ---------------------------
# Dataset Loader
# ---------------------------
class SyntheticDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []

        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    self.data.append(img_path)
                    self.labels.append(class_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx], img_path

# ---------------------------
# Helper Functions
# ---------------------------
def load_model(device, model_path):
    """Load the pre-trained CAE model."""
    cae = CAE().to(device)
    cae.load_state_dict(torch.load(model_path, map_location=device))
    cae.eval()
    return cae

def extract_latent_vectors(cae, data_loader, device):
    """Extract latent vectors for all samples using the CAE encoder."""
    latents, img_paths = [], []
    for x, _, paths in data_loader:
        x = x.to(device)
        with torch.no_grad():
            latent = cae.encoder(x).view(x.size(0), -1).cpu().numpy()
        latents.append(latent)
        img_paths.extend(paths)
    return np.vstack(latents), img_paths

def calculate_cluster_statistics(latent_vectors, cluster_labels):
    """Calculate cluster statistics, including the center and uncertainty metrics."""
    cluster_stats = {}
    cluster_pca_models = {}  # Store PCA models for each cluster
    unique_clusters = np.unique(cluster_labels)

    for cluster_id in unique_clusters:
        cluster_vectors = latent_vectors[cluster_labels == cluster_id]

        # Ensure enough samples for KDE by applying PCA if needed
        if cluster_vectors.shape[0] < cluster_vectors.shape[1]:
            print(f"[INFO] Applying PCA for cluster {cluster_id} due to low sample count.")
            pca = PCA(n_components=min(cluster_vectors.shape[0], cluster_vectors.shape[1]-1))
            cluster_vectors = pca.fit_transform(cluster_vectors)
            cluster_pca_models[cluster_id] = pca  # Save PCA model for later

        kde = gaussian_kde(cluster_vectors.T)
        density = kde(cluster_vectors.T)
        center_idx = np.argmax(density)
        cluster_center = cluster_vectors[center_idx]

        variances = np.mean((cluster_vectors - cluster_center) ** 2, axis=1)
        mean_variance = np.mean(variances)
        std_variance = np.std(variances)
        threshold = mean_variance + std_variance

        cluster_stats[cluster_id] = {
            "center": cluster_center,
            "mean_variance": mean_variance,
            "std_variance": std_variance,
            "threshold": threshold,
            "variances": variances
        }

    return cluster_stats, cluster_pca_models


    return cluster_stats

def save_uncertain_samples(cluster_stats, cluster_pca_models, latent_vectors, cluster_labels, img_paths, output_dir):
    """Save up to 5 uncertain samples per cluster."""
    for cluster_id, stats in cluster_stats.items():
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_vectors = latent_vectors[cluster_indices]
        cluster_paths = [img_paths[i] for i in cluster_indices]
        variances = stats["variances"]

        uncertain_indices = np.where(variances > stats["threshold"])[0]
        if len(uncertain_indices) == 0:
            continue

        uncertain_vectors = cluster_vectors[uncertain_indices]
        uncertain_paths = [cluster_paths[i] for i in uncertain_indices]

        # ✅ Apply the same PCA transformation used during clustering to uncertain vectors
        if cluster_id in cluster_pca_models:
            pca = cluster_pca_models[cluster_id]
            uncertain_vectors = pca.transform(uncertain_vectors)  # Reduce dimensionality

        # ✅ Now, uncertain_vectors and stats["center"] have the same shape!
        distances = [distance.euclidean(vec, stats["center"]) for vec in uncertain_vectors]
        sorted_indices = np.argsort(distances)[:5]  # Select 5 closest samples

        cluster_folder = os.path.join(output_dir, f"cluster_{cluster_id}")
        os.makedirs(cluster_folder, exist_ok=True)

        for idx in sorted_indices:
            img_path = uncertain_paths[idx]
            try:
                img = Image.open(img_path)
                img.save(os.path.join(cluster_folder, f"uncertain_sample_{os.path.basename(img_path)}"))
            except Exception as e:
                print(f"[ERROR] Could not save image {img_path}: {e}")


def plot_clusters(latent_vectors, cluster_labels, cluster_stats, output_path):
    """Visualize clusters with boundaries and highlight uncertain samples."""
    tsne = TSNE(n_components=2, random_state=42)
    reduced_vectors = tsne.fit_transform(latent_vectors)

    plt.figure(figsize=(10, 10))
    unique_clusters = np.unique(cluster_labels)

    for cluster_id in unique_clusters:
        cluster_points = reduced_vectors[cluster_labels == cluster_id]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_id}", alpha=0.6)

        # Draw convex hull for cluster boundaries
        if len(cluster_points) >= 3:
            try:
                hull = ConvexHull(cluster_points)
                for simplex in hull.simplices:
                    plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 'k-', linewidth=2)
            except scipy.spatial.qhull.QhullError:
                print(f"[WARNING] ConvexHull could not be created for cluster {cluster_id}. Skipping boundary visualization.")

    plt.legend()
    plt.title("Cluster Visualization with Uncertainty")
    plt.xlabel("TSNE Dimension 1")
    plt.ylabel("TSNE Dimension 2")
    plt.grid(True)
    plt.savefig(output_path, dpi=300)
    plt.close()

# ---------------------------
# Main Function
# ---------------------------
def main():
    device = set_device()
    cae = load_model(device, model_path)

    dataset = SyntheticDataset(root_dir=data_dir, transform=transforms.ToTensor())
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    latents, img_paths = extract_latent_vectors(cae, data_loader, device)
    cluster_labels = GaussianMixture(n_components=10, random_state=42).fit_predict(latents)

    # ✅ Updated function call to return PCA models
    cluster_stats, cluster_pca_models = calculate_cluster_statistics(latents, cluster_labels)

    # ✅ Pass PCA models to ensure consistency
    save_uncertain_samples(cluster_stats, cluster_pca_models, latents, cluster_labels, img_paths, output_dir)

    plot_clusters(latents, cluster_labels, cluster_stats, os.path.join(output_dir, "cluster_visualization.png"))

if __name__ == "__main__":
    main()
    