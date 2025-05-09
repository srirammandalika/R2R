# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from torch.utils.data import DataLoader

# def get_cifar10_data(batch_size, data_dir='/Users/srirammandalika/Downloads/Minor/CIFAR-10 data/cifar10/'):
#     transform_train = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomCrop(32, padding=4),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])

#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])

#     train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
#     test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     return train_loader, test_loader


# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from torch.utils.data import DataLoader

# def get_cifar10_data(batch_size, data_dir='/Users/srirammandalika/Downloads/Minor/CIFAR-10 data/cifar10/'):
#     """
#     Load CIFAR-10 training and testing data with appropriate transformations.
    
#     Args:
#         batch_size (int): Batch size for loading data.
#         data_dir (str): Directory where CIFAR-10 data is stored.

#     Returns:
#         tuple: (train_loader, test_loader) where train_loader and test_loader are DataLoader objects.
#     """
#     transform_train = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomCrop(32, padding=4),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])

#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])

#     train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
#     test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

#     return train_loader, test_loader

# def get_pseudo_labels(batch_size, pseudo_labels_path='/Users/srirammandalika/Downloads/Minor/Codes/data/pseudo_labels.npy'):
#     """
#     Load pseudo labels for use in the diffusion model.
    
#     Args:
#         batch_size (int): Batch size for loading data.
#         pseudo_labels_path (str): Path to the pseudo labels file.

#     Returns:
#         DataLoader: DataLoader object with pseudo labels.
#     """
#     import numpy as np
#     import torch

#     pseudo_labels = np.load(pseudo_labels_path)
#     pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long)
#     dataset = torch.utils.data.TensorDataset(pseudo_labels)
#     loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     return loader

##### With updated tasks generation


import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

def get_cifar10_data(batch_size, data_dir='/Users/srirammandalika/Downloads/Minor/CIFAR-10 data/cifar10/'):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader

def get_pseudo_labels(batch_size, pseudo_labels_path='/Users/srirammandalika/Downloads/Minor/Codes/data/pseudo_labels.npy'):
    import numpy as np
    import torch

    pseudo_labels = np.load(pseudo_labels_path)
    pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(pseudo_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader

def prepare_task_data(dataset, task_splits=5, samples_per_class=500):
    """
    Spliting CIFAR-10 dataset into multiple tasks.
    
    Args:
        dataset: The CIFAR-10 dataset.
        task_splits (int): Number of tasks to split the dataset into.
        samples_per_class (int): Number of samples per class in each task.

    Returns:
        list of Subset: List of Subset objects for each task.
    """
    classes_per_task = len(dataset.classes) // task_splits
    class_indices = {i: np.where(np.array(dataset.targets) == i)[0] for i in range(len(dataset.classes))}
    
    tasks = []
    for task in range(task_splits):
        task_classes = list(range(task * classes_per_task, (task + 1) * classes_per_task))
        task_indices = np.concatenate([class_indices[cls][:samples_per_class] for cls in task_classes])
        np.random.shuffle(task_indices)
        
        tasks.append(Subset(dataset, task_indices))
        
    return tasks
