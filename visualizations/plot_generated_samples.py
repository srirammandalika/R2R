# import os
# import torch
# import sys
# import matplotlib.pyplot as plt

# # Add the root directory of the project to sys.path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# sys.path.append(project_root)

# from models.diffusion_model import DiffusionModel

# def plot_samples(images, titles, save_dir="generated_samples"):
#     os.makedirs(save_dir, exist_ok=True)
#     plt.figure(figsize=(15, 10))
#     for i, (image, title) in enumerate(zip(images, titles)):
#         plt.subplot(1, len(images), i + 1)
#         plt.imshow(image)
#         plt.title(title)
#         plt.axis("off")
#         image.save(os.path.join(save_dir, f"{title}.png"))
#     plt.show()

# def main():
#     # Define device
#     device = "mps" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")

#     # Initialize the diffusion model
#     model = DiffusionModel(device=device)

#     # Example prompts to generate images
#     prompts = ["Car"]

#     # Generate images based on prompts
#     images = [model.generate(prompt) for prompt in prompts]

#     # Plot and save the generated samples
#     plot_samples(images, prompts)

# if __name__ == "__main__":
#     main()
