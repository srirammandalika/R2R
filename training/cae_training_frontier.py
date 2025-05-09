
###### SRC Code ######


import sys
import os
import time  # Added for timing epochs

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_root)

import json
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from models.cae_model import CAE
from utils.utils import set_device
from PIL import Image

# Paths
save_model_dir = '/Users/srirammandalika/Downloads/Minor/latents/'
save_tsne_dir = '/Users/srirammandalika/Downloads/Minor/data/tsne_plots/'
updated_model_path = '/Users/srirammandalika/Downloads/Minor/main weights/Loop 1/cae_main_v1.pth'  
reconstructed_images_dir = "/Users/srirammandalika/Downloads/Minor/latent images from clusters/Loop1/"

os.makedirs(save_model_dir, exist_ok=True)
os.makedirs(save_tsne_dir, exist_ok=True)
os.makedirs(reconstructed_images_dir, exist_ok=True)


def load_unlabelled_task_data(task_id, json_dir='/Users/srirammandalika/Downloads/Minor/data/CIFAR-10_data_json/', train_ratio=0.8, previous_test_data=None):
    """
    Load unlabelled data for each task and handle incremental test data.
    """
    # Load task data
    file_path = os.path.join(json_dir, f'cifar10_task{task_id}.json')
    try:
        with open(file_path, 'r') as f:
            task_info = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for Task {task_id}: {e}")
        return [], []

    data = []
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    # Process images
    for entry in task_info:
        try:
            img_path = entry.get("file_name", "")
            img = transform(Image.open(img_path).convert('RGB'))
            data.append(img.numpy())
        except Exception as e:
            print(f"Error processing entry {entry}: {e}")

    data = np.array(data)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    split_idx = int(train_ratio * len(data))

    # Split into training and testing
    train_data = data[indices[:split_idx]]
    test_data = data[indices[split_idx:]]

    # Handle cumulative test data across tasks
    if previous_test_data is None or len(previous_test_data) == 0:
        # Initialize test data for the first task
        cumulative_test_data = test_data
    else:
        # Ensure consistency in dimensions before concatenation
        previous_test_data = np.array(previous_test_data)
        if len(previous_test_data.shape) == 1:  # Fix if empty or 1-D array
            previous_test_data = previous_test_data.reshape(0, *test_data.shape[1:])
        cumulative_test_data = np.concatenate([previous_test_data, test_data], axis=0)

    return train_data, cumulative_test_data




def save_images(original_images, reconstructed_images, save_dir, task_id, cluster_id):
    """
    Save original and reconstructed images directly in the directory.
    """
    for idx, (original, reconstructed) in enumerate(zip(original_images, reconstructed_images)):
        # Convert tensors to PIL images
        original_img = transforms.ToPILImage()(original.cpu())
        reconstructed_img = transforms.ToPILImage()(torch.tensor(reconstructed).cpu())

        # Save images with task ID and cluster ID in filenames
        original_img.save(os.path.join(save_dir, f"task_{task_id}_cluster_{cluster_id}_original_{idx}.png"))
        reconstructed_img.save(os.path.join(save_dir, f"task_{task_id}_cluster_{cluster_id}_reconstructed_{idx}.png"))


def plot_tsne(latent_vectors, cluster_labels, save_path, title_suffix=""):
    """
    Plot TSNE visualization of latent space with boundaries and density points.
    """
    # Handle TSNE perplexity dynamically based on sample size
    n_samples = latent_vectors.shape[0]
    perplexity = max(2, min(30, n_samples - 1))  # Ensure valid perplexity range

    # Compute TSNE projection
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced_vectors = tsne.fit_transform(latent_vectors)

    # Create scatter plot
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(
        reduced_vectors[:, 0],
        reduced_vectors[:, 1],
        c=cluster_labels,
        cmap='tab10',
        alpha=0.6,
        s=40,
        edgecolors='k'
    )
    plt.colorbar(scatter, label="Cluster")
    plt.title(f"TSNE of Latent Space {title_suffix}")
    plt.xlabel("TSNE Dimension 1")
    plt.ylabel("TSNE Dimension 2")
    plt.grid(True)

    # Highlight boundaries and density points
    for cluster_id in np.unique(cluster_labels):
        cluster_points = reduced_vectors[cluster_labels == cluster_id]

        # Calculate density and find highest density point
        kde = gaussian_kde(cluster_points.T)
        density = kde(cluster_points.T)
        densest_point = cluster_points[np.argmax(density)]
        plt.scatter(densest_point[0], densest_point[1], c='red', s=150, marker='X', label=f"Cluster {cluster_id}")

        # Draw convex hull for boundaries
        if len(cluster_points) >= 3:
            hull = ConvexHull(cluster_points)
            for simplex in hull.simplices:
                plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 'k-', linewidth=2)

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"TSNE plot saved to {save_path}")



def train_cae_model(device, train_loader, num_epochs=25):
    """
    Train CAE with exponential learning rate decay.
    """
    cae = CAE().to(device)
    optimizer = torch.optim.Adam(cae.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(num_epochs):
        start_time = time.time()
        total_loss = 0

        for x in train_loader:
            x = x[0].to(device)
            recon = cae(x)
            loss = cae.loss_function(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Time: {epoch_time:.2f}s")

    return cae




def main():
    device = set_device()
    all_latent_vectors = []  # Store all latent vectors across tasks
    all_cluster_labels = []  # Store all cluster labels across tasks
    cluster_offset = 0  # Offset to manage cluster indices across tasks

    # Initialize CAE model
    cae = None

    # Maintain previous test data for incremental evaluation
    previous_test_data = []

    for task_id in range(1, 6):
        # Load data for the current task
        train_data, test_data = load_unlabelled_task_data(
            task_id,
            json_dir='/Users/srirammandalika/Downloads/Minor/data/CIFAR-10_data_json/',
            train_ratio=0.8,
            previous_test_data=previous_test_data
        )
        previous_test_data = test_data  # Update cumulative test data for next task

        # Create DataLoaders
        train_loader = DataLoader(
            TensorDataset(torch.tensor(train_data, dtype=torch.float32).to(device)),
            batch_size=32,
            shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(torch.tensor(test_data, dtype=torch.float32).to(device)),
            batch_size=32,
            shuffle=False
        )

        # Train or Fine-tune CAE
        if cae is None:  # Initial training
            cae = train_cae_model(device, train_loader)
        else:  # Fine-tune existing model incrementally
            cae = train_cae_model(device, train_loader)

        # Generate latent vectors for the current task (training data)
        latents = np.vstack([
            cae.encoder(x[0].to(device)).view(x[0].size(0), -1).detach().cpu().numpy()
            for x in train_loader
        ])

        # Perform clustering for the current task
        clusters = GaussianMixture(n_components=2).fit_predict(latents) + cluster_offset
        cluster_offset += 2  # Update offset for the next task

        # Append current task's latent vectors and cluster labels to the global lists
        all_latent_vectors.append(latents)
        all_cluster_labels.append(clusters)

        # Plot TSNE for the current task (training data)
        task_tsne_path = os.path.join(save_tsne_dir, f"task_{task_id}_latent_space.png")
        plot_tsne(latents, clusters, task_tsne_path, title_suffix=f"Task {task_id}")

        # --- Incremental Testing ---
        # Generate latent vectors for cumulative test data
        test_latents = np.vstack([
            cae.encoder(x[0].to(device)).view(x[0].size(0), -1).detach().cpu().numpy()
            for x in test_loader
        ])

        # Perform clustering for cumulative test data
        test_clusters = GaussianMixture(n_components=2).fit_predict(test_latents) + cluster_offset

        # Plot TSNE for cumulative test data
        test_tsne_path = os.path.join(save_tsne_dir, f"task_{task_id}_test_latent_space.png")
        plot_tsne(test_latents, test_clusters, test_tsne_path, title_suffix=f"Task {task_id} Test")

    # Combine all latent vectors and cluster labels across tasks
    all_latent_vectors = np.vstack(all_latent_vectors)
    all_cluster_labels = np.concatenate(all_cluster_labels)

    # Save the final CAE weights
    torch.save(cae.state_dict(), updated_model_path)
    print(f"CAE model weights saved to {updated_model_path}")

    # Plot combined TSNE for all tasks (training data)
    combined_tsne_path = os.path.join(save_tsne_dir, "all_tasks_latent_space.png")
    plot_tsne(all_latent_vectors, all_cluster_labels, combined_tsne_path, title_suffix="All Tasks")

    print("Combined TSNE plot saved successfully!")
    print("Incremental testing completed!")


if __name__ == "__main__":
    main()

