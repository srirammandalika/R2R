# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CAE(nn.Module):
#     def __init__(self):
#         super(CAE, self).__init__()
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 64 channels, 32x32
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0),       # 64 channels, 16x16
#             nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1), # 32 channels, 16x16
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0),       # 32 channels, 8x8
#         )
        
#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # 64 channels, 16x16
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1), # 3 channels, 32x32
#             nn.Sigmoid(),  # Normalize output to range [0, 1]
#         )

#     def forward(self, x):
#         encoded = self.encoder(x)  # Shape: [batch_size, 32, 8, 8]
#         decoded = self.decoder(encoded)  # Shape: [batch_size, 3, 32, 32]
#         return decoded

#     def loss_function(self, recon_x, x):
#         return F.mse_loss(recon_x, x)


###### Working Good ######

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CAE(nn.Module):
#     def __init__(self):
#         super(CAE, self).__init__()
        
#         # -------- ENCODER --------
#         self.encoder = nn.Sequential(
#             # Layer 1
#             nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),  # 128 channels, 32x32
#             nn.BatchNorm2d(128),  # Added BatchNorm for stability
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample -> 16x16
            
#             # Layer 2
#             nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # 64 channels, 16x16
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample -> 8x8
            
#             # Layer 3
#             nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),  # 32 channels, 8x8
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample -> 4x4
#         )
        
#         # -------- DECODER --------
#         self.decoder = nn.Sequential(
#             # Layer 1
#             nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample -> 8x8
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
            
#             # Layer 2
#             nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample -> 16x16
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
            
#             # Layer 3
#             nn.ConvTranspose2d(128, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample -> 32x32
#             nn.Sigmoid(),  # Normalize output to [0, 1]
#         )
    
#     def forward(self, x):
#         # Forward pass through encoder and decoder
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded

#     def loss_function(self, recon_x, x):
#         # MSE Loss for reconstruction
#         return F.mse_loss(recon_x, x)




######## New Model - 1 ########

import torch
import torch.nn as nn
import torch.nn.functional as F

class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        
        # -------- ENCODER --------
        self.encoder = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),  # 128 channels, 32x32
            nn.BatchNorm2d(128),  # Added BatchNorm for stability
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample -> 16x16
            
            # Layer 2
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # 64 channels, 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample -> 8x8
            
            # Layer 3
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),  # 32 channels, 8x8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample -> 4x4
        )
        
        # -------- DECODER --------
        self.decoder = nn.Sequential(
            # Layer 1
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample -> 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Layer 2
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample -> 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Layer 3
            nn.ConvTranspose2d(128, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample -> 32x32
            nn.Sigmoid(),  # Normalize output to [0, 1]
        )
    
    def forward(self, x):
        # Forward pass through encoder and decoder
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def loss_function(self, recon_x, x):
        # MSE Loss for reconstruction
        return F.mse_loss(recon_x, x)



######## New Model - 2 ######## - Don't use this one


# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class CAE(nn.Module):
#     def __init__(self):
#         super(CAE, self).__init__()
        
#         # -------- ENCODER --------
#         self.encoder = nn.Sequential(
#             # Layer 1
#             nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),  # 128 channels, 32x32
#             nn.BatchNorm2d(128),  # Batch Normalization
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample -> 16x16
#             nn.Dropout(0.2),  # Dropout Regularization
            
#             # Layer 2
#             nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # 64 channels, 16x16
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample -> 8x8
#             nn.Dropout(0.2),
            
#             # Layer 3
#             nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),  # 32 channels, 8x8
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample -> 4x4
            
#             # Layer 4 - Increase Latent Space Size
#             nn.Flatten(),
#             nn.Linear(32 * 4 * 4, 128),  # Latent size 128
#             nn.ReLU()
#         )
        
#         # -------- DECODER --------
#         self.decoder = nn.Sequential(
#             # Layer 1
#             nn.Linear(128, 32 * 4 * 4),  # Upscale latent size back
#             nn.ReLU(),
#             nn.Unflatten(1, (32, 4, 4)),
            
#             # Layer 2
#             nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample -> 8x8
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
            
#             # Layer 3
#             nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample -> 16x16
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
            
#             # Layer 4
#             nn.ConvTranspose2d(128, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample -> 32x32
#             nn.Sigmoid(),  # Normalize output to [0, 1]
#         )
    
#     def forward(self, x):
#         # Forward pass through encoder and decoder
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded

#     def loss_function(self, recon_x, x):
#         # MSE Loss for reconstruction
#         return F.mse_loss(recon_x, x)
