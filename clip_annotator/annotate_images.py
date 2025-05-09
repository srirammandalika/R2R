import os
import json
import torch
import numpy as np
from clip_model import load_clip_model, get_image_features, get_text_features
from PIL import Image

# Paths
image_dir = '/Users/srirammandalika/Downloads/Minor/latent images from clusters/Loop1'
cls_file = '/Users/srirammandalika/Downloads/Minor/data/CLS_names.json'
output_annotations = '/Users/srirammandalika/Downloads/Minor/clip_annotator/annotated_images.json'

# Device Setup
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

def load_word_labels(cls_file):
    """
    Load labels from the JSON file.
    """
    with open(cls_file, 'r') as f:
        data = json.load(f)
    return data['objects']


def annotate_images():
    # Load CLIP model
    model, preprocess = load_clip_model(device)

    # Load words from JSON
    labels = load_word_labels(cls_file)

    # Generate text features for words
    text_features = get_text_features(labels, model, device)

    # Process images
    annotations = {}
    for filename in os.listdir(image_dir):
        if filename.endswith(".png") and "original" in filename:  # Filter for original images
            image_path = os.path.join(image_dir, filename)

            # Get image features
            image_features = get_image_features(image_path, preprocess, model, device)

            # Calculate cosine similarity
            similarities = torch.cosine_similarity(image_features, text_features)
            best_match_index = similarities.argmax().item()
            best_label = labels[best_match_index]

            # Save annotations
            annotations[filename] = best_label
            print(f"Annotated {filename}: {best_label}")

    # Save annotations to JSON
    with open(output_annotations, 'w') as f:
        json.dump(annotations, f, indent=4)
    print(f"Annotations saved at {output_annotations}")


if __name__ == "__main__":
    annotate_images()
