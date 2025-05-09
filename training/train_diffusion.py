import sys
import os
import torch
import json

# Add the root directory of the project to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_root)

from models.diffusion_model import DiffusionModel

def load_mapping(task_id, mapping_dir='./mappings'):
    """Loads the mapping for the given task."""
    mapping_file = os.path.join(mapping_dir, f'mapping_task_{task_id}.json')
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    return mapping

def main():
    # Define device, prioritize mps over cuda
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize the diffusion model
    model = DiffusionModel(device=device)

    # Loop over tasks
    for task_id in range(1, 6):
        # Load the label mapping for the current task
        mapping = load_mapping(task_id)

        # Generate and save images based on the mapping
        for class_id, class_name in mapping.items():
            prompt = class_name
            image = model.generate(prompt)
            image.save(f"./data/synthetic_image_task_{task_id}_class_{class_id}.png")
            print(f"Generated image for Task {task_id}, Class {class_id} ({class_name})")

if __name__ == "__main__":
    main()
