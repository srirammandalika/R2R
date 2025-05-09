import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Paths
latent_dir = "/Users/srirammandalika/Downloads/Minor/latents/"
save_tsne_dir = "/Users/srirammandalika/Downloads/Minor/data/tsne_plots/"

# Ensure directories exist
os.makedirs(save_tsne_dir, exist_ok=True)


def load_latent_vectors():
    """Load latent vectors and cluster labels from saved .pt files."""
    all_latents = []
    all_clusters = []
    cluster_offset = 0  # For unique cluster IDs across tasks

    # Loop through tasks (1-5)
    for task_id in range(1, 6):
        latent_path = os.path.join(latent_dir, f'cae_task_{task_id}.pt')

        # Check if file exists
        if not os.path.exists(latent_path):
            print(f"Missing file for Task {task_id}: {latent_path}")
            continue

        # Load data
        try:
            data = torch.load(latent_path)

            # Ensure required keys exist
            if 'latent_vectors' in data and 'cluster_labels' in data:
                latents = data['latent_vectors']
                clusters = data['cluster_labels']

                # Convert tensors to NumPy arrays
                latents = latents.cpu().numpy() if isinstance(latents, torch.Tensor) else np.array(latents)
                clusters = clusters.cpu().numpy() if isinstance(clusters, torch.Tensor) else np.array(clusters)

                # Update cluster labels for unique coloring
                clusters += cluster_offset
                cluster_offset = clusters.max() + 1  # Update offset for next task

                # Append data
                all_latents.append(latents)
                all_clusters.append(clusters)
            else:
                print(f"Key error in {latent_path}: Missing 'latent_vectors' or 'cluster_labels'")
        except Exception as e:
            print(f"Error loading file {latent_path}: {e}")
            continue

    # Return combined results
    return (
        np.vstack(all_latents) if all_latents else np.array([]),
        np.hstack(all_clusters) if all_clusters else np.array([])
    )


def plot_combined_tsne(latent_vectors, cluster_labels, save_path, title):
    """Plot combined TSNE visualization for clusters with distinct colors."""
    # Handle TSNE perplexity dynamically
    perplexity = min(30, len(latent_vectors) - 1)
    perplexity = max(2, perplexity)

    # Perform TSNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    reduced_vectors = tsne.fit_transform(latent_vectors)

    # Number of unique clusters
    num_clusters = len(np.unique(cluster_labels))

    # Use 'tab20' colormap for distinct colors
    cmap = plt.get_cmap("tab20", num_clusters)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        reduced_vectors[:, 0],
        reduced_vectors[:, 1],
        c=cluster_labels,
        cmap=cmap,
        alpha=0.8,
        edgecolors='k',  # Add black border for clarity
        s=50  # Marker size
    )

    # Add Colorbar with Cluster IDs
    cbar = plt.colorbar(scatter, ticks=range(num_clusters))
    cbar.set_label('Cluster', fontsize=12)
    cbar.set_ticks(range(num_clusters))
    cbar.set_ticklabels([f'Cluster {i}' for i in range(num_clusters)])  # Labels for clusters

    # Plot Labels and Title
    plt.title(title, fontsize=16)
    plt.xlabel("TSNE Dimension 1", fontsize=12)
    plt.ylabel("TSNE Dimension 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save Plot
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"TSNE plot saved to {save_path}")


def main():
    # Load latent vectors and cluster labels
    latent_vectors, cluster_labels = load_latent_vectors()

    # Check if data loaded successfully
    if len(latent_vectors) == 0 or len(cluster_labels) == 0:
        print("No latent vectors or cluster labels found. Exiting...")
        return

    # Combined TSNE plot for all clusters
    plot_combined_tsne(
        latent_vectors,
        cluster_labels,
        save_path=os.path.join(save_tsne_dir, "ALL_Combined_clusters_tsne.png"),
        title="TSNE of Latent Space All Tasks"
    )


if __name__ == "__main__":
    main()
