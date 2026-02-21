# adain_sketch_train.py - Train Sketch Model Using Paired Data (Stable Version)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 features for better sketch details"""
    def __init__(self):
        super().__init__()
        # Load pretrained VGG16 and use first few layers
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
    
    def forward(self, output, target):
        """Compute L1 loss on VGG features"""
        output_features = self.vgg(output)
        target_features = self.vgg(target)
        return F.l1_loss(output_features, target_features)


class PairedSketchDataset(Dataset):
    """Paired dataset: content image + corresponding sketch image"""
    def __init__(self, root_dir, split='train', img_size=256):
        """
        Args:
            root_dir: Dataset root directory, should contain train/ and val/ subfolders
            split: 'train' or 'val'
            img_size: Image size
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        
        # Get paired files
        self.pairs = []
        content_dir = os.path.join(root_dir, split, 'content')
        sketch_dir = os.path.join(root_dir, split, 'sketch')
        
        # Check if directories exist
        if not os.path.exists(content_dir):
            print(f"❌ ERROR: Content directory does not exist: {content_dir}")
            return
        if not os.path.exists(sketch_dir):
            print(f"❌ ERROR: Sketch directory does not exist: {sketch_dir}")
            return
        
        # Get all content images
        content_files = sorted([f for f in os.listdir(content_dir) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"📂 Found {len(content_files)} content images")
        
        # Find corresponding sketch for each content image
        matched = 0
        for cf in content_files:
            found = False
            base_name = os.path.splitext(cf)[0]
            ext = os.path.splitext(cf)[1]
            
            # Try multiple possible sketch filenames
            possible_names = [
                cf,  # Same filename
                cf.replace('content', 'sketch'),  # content_01.jpg → sketch_01.jpg
                f"sketch_{cf}",  # content_01.jpg → sketch_content_01.jpg
                f"{base_name}_sketch{ext}",  # content_01.jpg → content_01_sketch.jpg
                base_name.replace('content', 'sketch') + ext,  # Replace prefix
            ]
            
            for sf in possible_names:
                sketch_path = os.path.join(sketch_dir, sf)
                if os.path.exists(sketch_path):
                    self.pairs.append({
                        'content': os.path.join(content_dir, cf),
                        'sketch': sketch_path
                    })
                    found = True
                    matched += 1
                    break
            
            if not found:
                print(f"⚠️ WARNING: Cannot find corresponding sketch for {cf}")
        
        print(f"✅ {split} set: {matched} image pairs matched successfully")
        
        if len(self.pairs) == 0:
            print(f"❌ ERROR: No paired images found in {split} set!")
            return
        
        # Data augmentation (for training set)
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Load images
        try:
            content_img = Image.open(pair['content']).convert('RGB')
            sketch_img = Image.open(pair['sketch']).convert('RGB')
        except Exception as e:
            print(f"❌ Failed to load image: {pair['content']} or {pair['sketch']}")
            print(f"Error message: {e}")
            # Return next image
            return self.__getitem__((idx + 1) % len(self.pairs))
        
        return self.transform(content_img), self.transform(sketch_img)


class ResidualBlock(nn.Module):
    """Residual block for decoder with instance normalization"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.in1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.in2 = nn.InstanceNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return F.relu(out + residual)


class AdaINModel(nn.Module):
    """AdaIN style transfer model with fixed VGG encoder and trainable decoder"""
    def __init__(self):
        super().__init__()
        
        # Fixed VGG19 encoder
        print("📦 Loading pretrained VGG19 encoder...")
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.encoder = nn.Sequential()
        
        # Use first 21 layers as encoder (up to relu4_1)
        for i, layer in enumerate(list(vgg.children())[:21]):
            self.encoder.add_module(str(i), layer)
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Trainable decoder with residual connections
        self.decoder = nn.Sequential(
            # Start from 512 channels
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        print(f"✅ Decoder parameter count: {sum(p.numel() for p in self.decoder.parameters()):,}")
    
    def calc_mean_std(self, feat, eps=1e-5):
        """Calculate feature mean and standard deviation"""
        size = feat.size()
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std
    
    def adaptive_instance_normalization(self, content_feat, style_feat):
        """Core AdaIN operation: match content feature statistics to style feature"""
        content_mean, content_std = self.calc_mean_std(content_feat)
        style_mean, style_std = self.calc_mean_std(style_feat)
        
        normalized_feat = (content_feat - content_mean) / content_std
        return normalized_feat * style_std + style_mean
    
    def forward(self, content, style):
        """
        Args:
            content: content image [B, 3, H, W]
            style: style image [B, 3, H, W] (sketch image here)
        Returns:
            stylized image [B, 3, H, W]
        """
        # Encode
        content_feat = self.encoder(content)
        style_feat = self.encoder(style)
        
        # AdaIN transfer
        t = self.adaptive_instance_normalization(content_feat, style_feat)
        
        # Decode
        return self.decoder(t)


def train_adain():
    print("=" * 60)
    print("🎨 ADAIN SKETCH TRAINING (STABLE VERSION)")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️ Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # ========== Path configuration ==========
    DATA_ROOT = "/content/drive/MyDrive/style-transfer-project/datasets"
    OUTPUT_DIR = "/content/drive/MyDrive/style-transfer-project/ai/outputs/adain_sketch"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    # ========== Training parameters ==========
    BATCH_SIZE = 4
    EPOCHS = 100
    LEARNING_RATE = 5e-5  # Lower learning rate for stable training
    IMAGE_SIZE = 256
    NUM_WORKERS = 2
    
    print(f"\n📊 Training parameters:")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Image Size: {IMAGE_SIZE}")
    
    # ========== Load dataset ==========
    print("\n📂 Loading dataset...")
    
    # Training set
    train_dataset = PairedSketchDataset(DATA_ROOT, 'train', IMAGE_SIZE)
    if len(train_dataset) == 0:
        print("❌ Training set is empty, cannot train!")
        return
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        drop_last=True
    )
    
    # Validation set
    val_dataset = PairedSketchDataset(DATA_ROOT, 'val', IMAGE_SIZE)
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=NUM_WORKERS
        )
        print(f"✅ Validation set: {len(val_dataset)} image pairs")
    else:
        print("⚠️ Validation set is empty, using training set for validation")
        val_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=NUM_WORKERS
        )
    
    print(f"✅ Training set: {len(train_dataset)} image pairs")
    print(f"✅ Iterations per epoch: {len(train_loader)}")
    
    # ========== Create model ==========
    print("\n🤖 Creating model...")
    model = AdaINModel().to(device)
    
    # Optimize decoder only
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Loss functions
    mse_loss = nn.MSELoss()
    perceptual_loss = PerceptualLoss().to(device)
    
    # ========== Training ==========
    print(f"\n🚀 Start training for {EPOCHS} epochs...")
    
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
        for content, sketch in pbar:
            content = content.to(device)
            sketch = sketch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(content, sketch)
            
            # Combined loss: MSE + perceptual loss (with small weight)
            mse_loss_val = mse_loss(output, sketch)
            perceptual_loss_val = perceptual_loss(output, sketch)
            loss = mse_loss_val + 0.05 * perceptual_loss_val  # Perceptual loss helps with details
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        if val_loader is not None:
            with torch.no_grad():
                for content, sketch in val_loader:
                    content = content.to(device)
                    sketch = sketch.to(device)
                    output = model(content, sketch)
                    val_loss += mse_loss(output, sketch).item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # Update learning rate based on validation loss
            scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.decoder.state_dict(), 
                          os.path.join(OUTPUT_DIR, 'best_model.pth'))
                print(f"\n💾 Saved best model (val_loss: {avg_val_loss:.4f})")
            
            print(f"\n📊 Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        else:
            print(f"\n📊 Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")
        
        # Save checkpoint and visualize every 10 epochs
        if (epoch + 1) % 10 == 0:
            # Save checkpoint
            torch.save({
                'epoch': epoch + 1,
                'decoder_state_dict': model.decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss if val_loader else avg_train_loss,
            }, os.path.join(OUTPUT_DIR, f'checkpoint_epoch_{epoch+1}.pth'))
            
            # Visualize results
            model.eval()
            with torch.no_grad():
                try:
                    if val_loader:
                        content_val, sketch_val = next(iter(val_loader))
                    else:
                        content_val, sketch_val = next(iter(train_loader))
                except StopIteration:
                    continue
                
                content_val = content_val[:4].to(device)
                sketch_val = sketch_val[:4].to(device)
                output_val = model(content_val, sketch_val)
                
                # Denormalization function
                def denorm(t):
                    return (t.cpu().numpy().transpose(1, 2, 0) + 1) / 2
                
                fig, axes = plt.subplots(min(4, len(content_val)), 3, figsize=(12, 14))
                if len(content_val) == 1:
                    axes = axes.reshape(1, -1)
                
                for i in range(min(4, len(content_val))):
                    # Content image
                    axes[i, 0].imshow(np.clip(denorm(content_val[i]), 0, 1))
                    axes[i, 0].set_title('Content', fontsize=12)
                    axes[i, 0].axis('off')
                    
                    # Generated sketch
                    axes[i, 1].imshow(np.clip(denorm(output_val[i]), 0, 1))
                    axes[i, 1].set_title('Generated Sketch', fontsize=12, color='red' if i==0 else 'black')
                    axes[i, 1].axis('off')
                    
                    # Target sketch
                    axes[i, 2].imshow(np.clip(denorm(sketch_val[i]), 0, 1))
                    axes[i, 2].set_title('Target Sketch', fontsize=12)
                    axes[i, 2].axis('off')
                
                plt.suptitle(f'Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}', fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, f'result_epoch_{epoch+1}.png'), dpi=150, bbox_inches='tight')
                plt.show()
    
    # ========== Save final model ==========
    final_path = os.path.join(OUTPUT_DIR, 'adain_sketch_final.pth')
    torch.save(model.decoder.state_dict(), final_path)
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    if val_losses:
        plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curve.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Training completed!")
    print(f"💾 Final model saved to: {final_path}")
    print(f"💾 Best model saved to: {os.path.join(OUTPUT_DIR, 'best_model.pth')}")
    print(f"📈 Loss curve saved to: {os.path.join(OUTPUT_DIR, 'loss_curve.png')}")
    print("=" * 60)


if __name__ == "__main__":
    train_adain()