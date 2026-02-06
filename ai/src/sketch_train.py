# sketch_train.py - Train sketch style model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import StyleTransferModel, count_parameters

class SketchDataset(Dataset):
    def __init__(self, content_dir, sketch_dir, img_size=256):
        self.content_dir = content_dir
        self.sketch_dir = sketch_dir
        
        # Get image paths
        self.content_paths = [os.path.join(content_dir, f) for f in os.listdir(content_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.sketch_paths = [os.path.join(sketch_dir, f) for f in os.listdir(sketch_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Content images: {len(self.content_paths)}")
        print(f"Sketch images: {len(self.sketch_paths)}")
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.content_paths) * 5
    
    def __getitem__(self, idx):
        content_idx = idx % len(self.content_paths)
        sketch_idx = np.random.randint(0, len(self.sketch_paths))
        
        content_img = Image.open(self.content_paths[content_idx]).convert('RGB')
        sketch_img = Image.open(self.sketch_paths[sketch_idx]).convert('RGB')
        
        return self.transform(content_img), self.transform(sketch_img)

def train_sketch():
    print("=" * 60)
    print("SKETCH STYLE TRANSFER TRAINING")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Paths
    CONTENT_DIR = "../../datasets/content"
    SKETCH_DIR = "../../datasets/styles/sketch"
    OUTPUT_DIR = "../outputs/sketch"
    
    # Training parameters
    IMAGE_SIZE = 256
    BATCH_SIZE = 4
    EPOCHS = 30  # Start with 30 epochs
    LEARNING_RATE = 0.001
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Content dir: {CONTENT_DIR}")
    print(f"Sketch dir: {SKETCH_DIR}")
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = SketchDataset(CONTENT_DIR, SKETCH_DIR, IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Create model
    print("\nCreating model...")
    model = StyleTransferModel(num_channels=32, num_residual_blocks=5)
    model = model.to(device)
    print(f"Parameters: {count_parameters(model):,}")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # Training loop
    print(f"\nTraining for {EPOCHS} epochs...")
    print("Estimated time: 20-40 minutes")
    
    losses = []
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for content, sketch in pbar:
            content = content.to(device)
            sketch = sketch.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            output = model(content)
            loss = criterion(output, sketch)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        # Save every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = os.path.join(OUTPUT_DIR, f'epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
            }, checkpoint)
            print(f"Saved: {checkpoint}")
    
    # Save final model
    final_path = os.path.join(OUTPUT_DIR, 'sketch_model.pth')
    torch.save(model.state_dict(), final_path)
    
    # Plot loss
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    loss_plot = os.path.join(OUTPUT_DIR, 'loss_plot.png')
    plt.savefig(loss_plot)
    plt.show()
    
    print(f"\n✅ Training complete!")
    print(f"Final model: {final_path}")
    print(f"Loss plot: {loss_plot}")
    print("=" * 60)

if __name__ == "__main__":
    train_sketch()
