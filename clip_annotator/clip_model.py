import torch
from PIL import Image
from torchvision import transforms
import open_clip
print(open_clip.__version__)


def load_clip_model(device):
    """
    Load the CLIP model and processor.
    """
    # OpenCLIP model setup
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
    return model, preprocess


def get_image_features(image_path, preprocess, model, device):
    """
    Extract image features using CLIP.
    """
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features


def get_text_features(labels, model, device):
    """
    Extract text features for label names.
    """
    text_inputs = open_clip.tokenize(labels).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    return text_features
