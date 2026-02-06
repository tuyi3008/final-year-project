# model.py - Neural Style Transfer Model

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block with instance normalization"""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
        )
    
    def forward(self, x):
        return x + self.block(x)

class StyleTransferModel(nn.Module):
    """Main style transfer model"""
    def __init__(self, num_channels=32, num_residual_blocks=5):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, num_channels, 3, padding=1),
            nn.InstanceNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(num_channels) for _ in range(num_residual_blocks)
        ])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(num_channels, 3, 3, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def forward(self, x):
        features = self.encoder(x)
        transformed = self.res_blocks(features)
        return self.decoder(transformed)

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
