# import torch
# import os
# import json

# # Assuming you have the class names for the teacher model's pseudo labels
# class_names = {
#     0: "airplane",
#     1: "automobile",
#     2: "bird",
#     3: "cat",
#     4: "deer",
#     5: "dog",
#     6: "frog",
#     7: "horse",
#     8: "ship",
#     9: "truck"
# }

# def load_weak_classes(task_id, file_dir='./Support Files'):
#     """Load the weak classes from the .pt files."""
#     file_path = os.path.join(file_dir, f'original_{task_id}.pt')
#     return torch.load(file_path)

# def create_mapping(task_id, persistent_mapping, file_dir='./Support Files', save_dir='./mappings'):
#     """Create a mapping for the current task, ensuring consistent class assignments."""
#     weak_classes = load_weak_classes(task_id, file_dir)

#     # Maintain persistent mapping across tasks
#     current_task_mapping = {}
#     class_counter = len(persistent_mapping)  # Start assigning new classes from the next available index

#     for cls, accuracy in weak_classes.items():
#         # If cls is a string (like "automobile"), get its corresponding index from class_names
#         cls_index = [index for index, name in class_names.items() if name == cls]
        
#         if not cls_index:
#             raise ValueError(f"Class '{cls}' not found in class_names.")
        
#         cls_index = cls_index[0]
#         actual_class_label = class_names[cls_index]

#         # Check if this class label has already been mapped
#         if actual_class_label in persistent_mapping.values():
#             # Find the existing class index for this label
#             existing_class_index = list(persistent_mapping.keys())[list(persistent_mapping.values()).index(actual_class_label)]
#             current_task_mapping[existing_class_index] = actual_class_label
#         else:
#             # Assign a new class index
#             new_class_index = f'class_{class_counter}'
#             persistent_mapping[new_class_index] = actual_class_label
#             current_task_mapping[new_class_index] = actual_class_label
#             class_counter += 1

#     # Save the current task mapping to a JSON file
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, f'mapping_task_{task_id}.json')
#     with open(save_path, 'w') as f:
#         json.dump(current_task_mapping, f, indent=4)
#     print(f"Saved mapping for Task {task_id} to {save_path}")

#     return persistent_mapping

# def main():
#     persistent_mapping = {}  # Initialize an empty dictionary to keep track of all mapped classes

#     for task_id in range(1, 6):
#         print(f"\nCreating mapping for Task {task_id}")
#         persistent_mapping = create_mapping(task_id, persistent_mapping)

# if __name__ == "__main__":
#     main()



import torch
import os
import json
import numpy as np

# Assuming you have the class names for the teacher model's pseudo labels
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

def load_weak_classes(task_id, file_dir='./Support Files'):
    """Load the weak classes from the .pt files."""
    file_path = os.path.join(file_dir, f'original_{task_id}.pt')
    return torch.load(file_path)

def load_pseudo_labels(label_file='./visualizations/Pretrained_labels.npy', pred_file='./visualizations/Pretrained_predictions.npy'):
    """Load the pseudo labels and predictions from the saved .npy files."""
    labels = np.load(label_file)
    preds = np.load(pred_file)
    return labels, preds

def create_mapping(task_id, persistent_mapping, labels, preds, file_dir='./Support Files', save_dir='./mappings'):
    """Create a mapping for the current task, ensuring consistent class assignments."""
    weak_classes = load_weak_classes(task_id, file_dir)
    current_task_mapping = {}
    class_counter = len(persistent_mapping)  # Start assigning new classes from the next available index

    for weak_class_str in weak_classes:
        # Map the weak class string to its corresponding numeric index
        weak_class_index = None
        for idx, name in class_names.items():
            if name == weak_class_str:
                weak_class_index = idx
                break
        
        if weak_class_index is None:
            raise ValueError(f"Class '{weak_class_str}' not found in class_names.")
        
        # Find the corresponding pseudo label from the teacher model
        indices = np.where(preds == weak_class_index)
        if len(indices[0]) == 0:
            raise ValueError(f"No pseudo labels found for weak class {weak_class_str} (index {weak_class_index}).")
        pseudo_label = labels[indices[0][0]]

        actual_class_label = class_names[pseudo_label]

        # Check if this class label has already been mapped
        if actual_class_label in persistent_mapping.values():
            # Find the existing class index for this label
            existing_class_index = list(persistent_mapping.keys())[list(persistent_mapping.values()).index(actual_class_label)]
            current_task_mapping[existing_class_index] = actual_class_label
        else:
            # Assign a new class index
            new_class_index = f'class_{class_counter}'
            persistent_mapping[new_class_index] = actual_class_label
            current_task_mapping[new_class_index] = actual_class_label
            class_counter += 1

    # Save the current task mapping to a JSON file
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'mapping_task_{task_id}.json')
    with open(save_path, 'w') as f:
        json.dump(current_task_mapping, f, indent=4)
    print(f"Saved mapping for Task {task_id} to {save_path}")

    return persistent_mapping

def main():
    persistent_mapping = {}  # Initialize an empty dictionary to keep track of all mapped classes

    # Load the pseudo labels and predictions
    labels, preds = load_pseudo_labels()

    for task_id in range(1, 6):
        print(f"\nCreating mapping for Task {task_id}")
        persistent_mapping = create_mapping(task_id, persistent_mapping, labels, preds)

if __name__ == "__main__":
    main()
