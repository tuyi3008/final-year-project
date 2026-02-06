# train.py - Style Transfer Training Script

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

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
        return len(self.content_paths) * 10  # Increase dataset size
    
    def __getitem__(self, idx):
        content_idx = idx % len(self.content_paths)
        style_idx = np.random.randint(0, len(self.style_paths))
        
        content_img = Image.open(self.content_paths[content_idx]).convert('RGB')
        style_img = Image.open(self.style_paths[style_idx]).convert('RGB')
        
        return self.transform(content_img), self.transform(style_img)

def train_model():
    """Main training function"""
    print("=" * 60)
    print("STARTING STYLE TRANSFER TRAINING")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ========== CONFIGURATION ==========
    # YOU NEED TO SET THESE PATHS!
    CONTENT_DIR = "/content/drive/MyDrive/style-transfer-project/datasets/content"
    STYLE_DIR = "/content/drive/MyDrive/style-transfer-project/datasets/styles/sketch"
    OUTPUT_DIR = "/content/drive/MyDrive/style-transfer-project/ai/outputs"
    
    # Training parameters
    IMAGE_SIZE = 256
    BATCH_SIZE = 4
    EPOCHS = 2  # Start with 10 epochs for testing
    LEARNING_RATE = 0.001
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Content directory: {CONTENT_DIR}")
    print(f"Style directory: {STYLE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # ========== CREATE DATASET ==========
    print("\nCreating dataset...")
    dataset = StyleTransferDataset(CONTENT_DIR, STYLE_DIR, IMAGE_SIZE)
    
    if len(dataset) == 0:
        print("ERROR: No images found in directories!")
        print(f"Please add images to:")
        print(f"  {CONTENT_DIR}")
        print(f"  {STYLE_DIR}")
        return
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # ========== CREATE MODEL ==========
    print("\nCreating model...")
    model = StyleTransferModel(num_channels=32, num_residual_blocks=5)
    model = model.to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # ========== SETUP TRAINING ==========
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Simple loss function
    criterion = nn.MSELoss()
    
    # ========== TRAINING LOOP ==========
    print(f"\nStarting training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for content_batch, style_batch in pbar:
            # Move to device
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            stylized = model(content_batch)
            
            # Compute loss (simple style transfer loss)
            content_loss = criterion(stylized, content_batch) * 0.5
            style_loss = criterion(stylized, style_batch) * 0.5
            loss = content_loss + style_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update progress
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Print epoch summary
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(OUTPUT_DIR, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # ========== SAVE FINAL MODEL ==========
    final_model_path = os.path.join(OUTPUT_DIR, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✅ Training completed!")
    print(f"Final model saved: {final_model_path}")
    print("=" * 60)

if __name__ == "__main__":
    train_model()
