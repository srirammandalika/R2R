import sys
import os
import torch
import numpy as np
from torchvision import transforms

# Ensure the models directory is on the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_root)
models_dir = os.path.join(project_root, 'models')
sys.path.append(models_dir)

from models.diffusion_model import DiffusionModel

def load_mapped_weak_classes(task_id, file_dir='./Support Files'):
    """Load the mapped weaker classes from the .pt files."""
    file_path = os.path.join(file_dir, f'original_{task_id}.pt')
    return torch.load(file_path)

def generate_synthetic_images(diffusion_model, class_name, num_samples=500):
    """Generate synthetic images for a given class using the generative model."""
    synthetic_data = []
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    
    for _ in range(num_samples):
        prompt = f"{class_name}"
        generated_image = diffusion_model.generate(prompt)
        synthetic_data.append(transform(generated_image).numpy())
    
    return torch.tensor(np.array(synthetic_data))

def main():
    device = "mps"  # Update based on your device
    diffusion_model = DiffusionModel(device=device)

    for task_id in range(1, 6):  # For each task (1 to 5)
        print(f"\n--- Generating Synthetic Data for Task {task_id} ---")
        mapped_weak_classes = load_mapped_weak_classes(task_id)

        if not mapped_weak_classes:
            print(f"No weaker classes found for Task {task_id}. Skipping...")
            continue

        for class_name, _ in mapped_weak_classes.items():
            print(f"Generating {500} samples for class: {class_name}")
            # Synthetic data generation (commented out actual saving for debugging)
            # synthetic_data = generate_synthetic_images(diffusion_model, class_name, num_samples=500)

if __name__ == "__main__":
    main()






























# import sys
# import os
# # Add the root directory of the project to sys.path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# sys.path.append(project_root)

# import torch
# import numpy as np
# from torchvision import transforms
# from models.diffusion_model import DiffusionModel

# def load_mapped_weak_classes(task_id, file_dir='./Support Files'):
#     """Load the mapped weaker classes from the .pt files."""
#     file_path = os.path.join(file_dir, f'original_{task_id}.pt')
#     return torch.load(file_path)

# def generate_synthetic_images(diffusion_model, class_name, num_samples=500):
#     """Generate synthetic images for a given class using the generative model."""
#     synthetic_data = []
#     transform = transforms.Compose([
#         transforms.Resize((32, 32)),
#         transforms.ToTensor()
#     ])
    
#     for _ in range(num_samples):
#         prompt = f"{class_name}"
#         generated_image = diffusion_model.generate(prompt)
#         synthetic_data.append(transform(generated_image).numpy())
    
#     return torch.tensor(np.array(synthetic_data))

# def save_synthetic_data(synthetic_data, synthetic_labels, task_id, save_dir='./SyntheticData'):
#     """Save synthetic data and labels to disk."""
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     data_path = os.path.join(save_dir, f'synthetic_data_task_{task_id}.pt')
#     labels_path = os.path.join(save_dir, f'synthetic_labels_task_{task_id}.pt')
    
#     torch.save(synthetic_data, data_path)
#     torch.save(synthetic_labels, labels_path)
#     print(f"Saved synthetic data for Task {task_id} at {data_path}")

# def main():
#     device = "mps"  # Update based on your device
#     diffusion_model = DiffusionModel(device=device)

#     for task_id in range(1, 6):  # For each task (1 to 5)
#         print(f"\n--- Generating Synthetic Data for Task {task_id} ---")
#         mapped_weak_classes = load_mapped_weak_classes(task_id)

#         if not mapped_weak_classes:
#             print(f"No weaker classes found for Task {task_id}. Skipping...")
#             continue

#         all_synthetic_data = []
#         all_synthetic_labels = []
        
#         for class_name, _ in mapped_weak_classes.items():
#             print(f"Generating {500} samples for class: {class_name}")
#             synthetic_data = generate_synthetic_images(diffusion_model, class_name, num_samples=500)
#             synthetic_labels = torch.full((500,), list(mapped_weak_classes.keys()).index(class_name))  # Assign index as label
            
#             all_synthetic_data.append(synthetic_data)
#             all_synthetic_labels.append(synthetic_labels)
        
#         if all_synthetic_data:  # Check if synthetic data was generated
#             # Combine all synthetic data and labels
#             all_synthetic_data = torch.cat(all_synthetic_data)
#             all_synthetic_labels = torch.cat(all_synthetic_labels)

#             # Save synthetic data and labels
#             save_synthetic_data(all_synthetic_data, all_synthetic_labels, task_id)
#         else:
#             print(f"No synthetic data generated for Task {task_id}.")

# if __name__ == "__main__":
#     main()
