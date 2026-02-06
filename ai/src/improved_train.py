# improved_train.py - With proper style transfer loss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
from PIL import Image
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import our model
from model import StyleTransferModel, count_parameters

class StyleTransferDataset(Dataset):
    """Dataset for style transfer training"""
    def __init__(self, content_dir, style_dir, img_size=256):
        self.content_dir = content_dir
        self.style_dir = style_dir
        self.img_size = img_size
        
        # Get image paths
        self.content_paths = self._get_image_paths(content_dir)
        self.style_paths = self._get_image_paths(style_dir)
        
        print(f"Content images: {len(self.content_paths)}")
        print(f"Style images: {len(self.style_paths)}")
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def _get_image_paths(self, directory):
        """Get all image paths from directory"""
        if not os.path.exists(directory):
            print(f"Warning: Directory not found: {directory}")
            return []
        
        extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        paths = []
        for file in os.listdir(directory):
            if file.lower().endswith(extensions):
                paths.append(os.path.join(directory, file))
        
        return paths
    
    def __len__(self):
        return len(self.content_paths) * 10
    
    def __getitem__(self, idx):
        content_idx = idx % len(self.content_paths)
        style_idx = np.random.randint(0, len(self.style_paths))
        
        content_img = Image.open(self.content_paths[content_idx]).convert('RGB')
        style_img = Image.open(self.content_paths[style_idx]).convert('RGB')  # 先用内容图片作为风格
        
        return self.transform(content_img), self.transform(style_img)

class StyleLoss(nn.Module):
    """Proper style transfer loss using Gram matrices"""
    def __init__(self):
        super().__init__()
        # Use VGG19 for feature extraction
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        # Extract features from these layers
        self.layers = {
            '0': 'conv1_1',
            '5': 'conv2_1', 
            '10': 'conv3_1',
            '19': 'conv4_1',
            '28': 'conv5_1'
        }
        
        self.model = nn.Sequential()
        self.feature_maps = {}
        
        # Build feature extractor
        for i, layer in enumerate(vgg):
            if isinstance(layer, nn.Conv2d):
                layer = nn.Conv2d(layer.in_channels, layer.out_channels,
                                 kernel_size=layer.kernel_size,
                                 stride=layer.stride,
                                 padding=layer.padding)
                layer.weight.data = vgg[i].weight.data.clone()
                layer.bias.data = vgg[i].bias.data.clone()
            
            self.model.add_module(str(i), layer)
            
            if str(i) in self.layers:
                def hook(module, input, output, key=str(i)):
                    self.feature_maps[key] = output
                layer.register_forward_hook(hook)
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def gram_matrix(self, features):
        """Compute Gram matrix for style features"""
        b, c, h, w = features.size()
        features = features.view(b * c, h * w)
        gram = torch.mm(features, features.t())
        return gram.div(b * c * h * w)
    
    def forward(self, content, style, generated):
        """Compute style loss"""
        self.model(content)
        content_features = {k: v.clone() for k, v in self.feature_maps.items()}
        
        self.model(style)
        style_features = {k: v.clone() for k, v in self.feature_maps.items()}
        
        self.model(generated)
        generated_features = {k: v.clone() for k, v in self.feature_maps.items()}
        
        # Content loss
        content_loss = 0
        for layer in ['conv4_1']:  # Use deeper layer for content
            key = list(self.layers.keys())[list(self.layers.values()).index(layer)]
            content_loss += torch.mean((content_features[key] - generated_features[key]) ** 2)
        
        # Style loss
        style_loss = 0
        for layer in self.layers.values():
            key = list(self.layers.keys())[list(self.layers.values()).index(layer)]
            gram_style = self.gram_matrix(style_features[key])
            gram_generated = self.gram_matrix(generated_features[key])
            style_loss += torch.mean((gram_style - gram_generated) ** 2)
        
        return content_loss, style_loss

def train_model():
    """Main training function with style loss"""
    print("=" * 60)
    print("ADVANCED STYLE TRANSFER TRAINING")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configuration
    CONTENT_DIR = "../../datasets/content"
    STYLE_DIR = "../../datasets/styles/sketch"
    OUTPUT_DIR = "../outputs"
    
    IMAGE_SIZE = 256
    BATCH_SIZE = 2  # Smaller batch for VGG features
    EPOCHS = 5      # Still small for testing
    LEARNING_RATE = 0.001
    
    # Content/Style loss weights
    CONTENT_WEIGHT = 1
    STYLE_WEIGHT = 1e6
    
    # Create directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Content directory: {CONTENT_DIR}")
    print(f"Style directory: {STYLE_DIR}")
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = StyleTransferDataset(CONTENT_DIR, STYLE_DIR, IMAGE_SIZE)
    
    if len(dataset) == 0:
        print("ERROR: No images found!")
        return
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Create models
    print("\nCreating models...")
    model = StyleTransferModel(num_channels=32, num_residual_blocks=5)
    model = model.to(device)
    
    style_loss_fn = StyleLoss().to(device)
    
    print(f"Style transfer model parameters: {count_parameters(model):,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print(f"\nStarting training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_content_loss = 0
        total_style_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for content_batch, style_batch in pbar:
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            stylized = model(content_batch)
            
            # Compute losses
            content_loss, style_loss = style_loss_fn(
                content_batch, style_batch, stylized
            )
            
            total_loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Update progress
            total_content_loss += content_loss.item()
            total_style_loss += style_loss.item()
            
            pbar.set_postfix({
                'content': f'{content_loss.item():.4f}',
                'style': f'{style_loss.item():.4f}'
            })
        
        # Print epoch summary
        avg_content = total_content_loss / len(dataloader)
        avg_style = total_style_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Content Loss = {avg_content:.4f}, Style Loss = {avg_style:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 2 == 0:
            checkpoint_path = os.path.join(OUTPUT_DIR, f'advanced_checkpoint_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'content_loss': avg_content,
                'style_loss': avg_style,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(OUTPUT_DIR, 'advanced_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ Advanced training completed!")
    print(f"Model saved: {final_path}")
    print("=" * 60)

if __name__ == "__main__":
    train_model()
