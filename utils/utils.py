import torch
import numpy as np
import os

def set_device():
    """Set the device to MPS if available, otherwise CPU."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def save_model(model, path):
    """Save the model to the specified path."""
    torch.save(model.state_dict(), path)

def save_pseudo_labels(pseudo_labels, path):
    """Save pseudo labels to the specified path."""
    np.save(path, pseudo_labels)

def get_pseudo_labels(batch_size, data_dir='./data/pseudo_labels/'):
    """Load pseudo labels for use in the diffusion model."""
    labels = np.load(os.path.join(data_dir, 'pseudo_labels.npy'))
    labels = torch.tensor(labels, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader
