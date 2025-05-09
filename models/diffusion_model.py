from diffusers import StableDiffusionPipeline
import torch

class DiffusionModel:
    def __init__(self, model_name="CompVis/stable-diffusion-v1-4", device="mps"):
        # Determine the correct device
        self.device = device if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32  # Use float32 for mps
        
        # Load the stable diffusion model with the appropriate dtype
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name, torch_dtype=dtype
        )
        self.pipe = self.pipe.to(self.device)
    
    def generate(self, prompt):
        # Generate an image based on the prompt
        if self.device == "cuda":
            with torch.autocast(self.device):
                image = self.pipe(prompt).images[0]
        else:
            # For 'mps' or 'cpu', run without autocast
            image = self.pipe(prompt).images[0]
        return image