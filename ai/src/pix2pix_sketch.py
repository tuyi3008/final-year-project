# pix2pix_sketch.py - Complete Pix2Pix for Sketch Generation (Stable Version)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# ==================== Dataset ====================
class PairedSketchDataset(Dataset):
    """Paired dataset: content image + corresponding sketch image"""
    def __init__(self, root_dir, split='train', img_size=256):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        
        # Get paired files
        self.pairs = []
        content_dir = os.path.join(root_dir, split, 'content')
        sketch_dir = os.path.join(root_dir, split, 'sketch')
        
        if not os.path.exists(content_dir) or not os.path.exists(sketch_dir):
            print(f"❌ ERROR: Directory does not exist")
            return
        
        # Get all content images
        content_files = sorted([f for f in os.listdir(content_dir) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"📂 Found {len(content_files)} content images")
        
        # Match by filename
        matched = 0
        for cf in content_files:
            sketch_path = os.path.join(sketch_dir, cf)
            if os.path.exists(sketch_path):
                self.pairs.append({
                    'content': os.path.join(content_dir, cf),
                    'sketch': sketch_path
                })
                matched += 1
            else:
                print(f"⚠️ WARNING: Cannot find corresponding sketch for {cf}")
        
        print(f"✅ {split} set: {matched} image pairs matched")
        
        # Transform
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
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
        content_img = Image.open(pair['content']).convert('RGB')
        sketch_img = Image.open(pair['sketch']).convert('RGB')
        return self.transform(content_img), self.transform(sketch_img)


# ==================== U-Net Generator ====================
class UNet(nn.Module):
    """U-Net architecture for image-to-image translation"""
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # Encoder (downsampling)
        self.enc1 = self.conv_block(in_channels, 64, batchnorm=False)  # 256x256
        self.enc2 = self.conv_block(64, 128)    # 128x128
        self.enc3 = self.conv_block(128, 256)   # 64x64
        self.enc4 = self.conv_block(256, 512)   # 32x32
        self.enc5 = self.conv_block(512, 512)   # 16x16
        self.enc6 = self.conv_block(512, 512)   # 8x8
        self.enc7 = self.conv_block(512, 512)   # 4x4
        self.enc8 = self.conv_block(512, 512, batchnorm=False)  # 2x2
        
        # Decoder (upsampling with skip connections)
        self.dec8 = self.upconv_block(512, 512, dropout=True)    # 2x2 → 4x4
        self.dec7 = self.upconv_block(1024, 512, dropout=True)   # 4x4 → 8x8
        self.dec6 = self.upconv_block(1024, 512, dropout=True)   # 8x8 → 16x16
        self.dec5 = self.upconv_block(1024, 512)                 # 16x16 → 32x32
        self.dec4 = self.upconv_block(1024, 256)                 # 32x32 → 64x64
        self.dec3 = self.upconv_block(512, 128)                  # 64x64 → 128x128
        self.dec2 = self.upconv_block(256, 64)                   # 128x128 → 256x256
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
    def conv_block(self, in_channels, out_channels, batchnorm=True):
        """Convolution block for encoder"""
        layers = [
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)
    
    def upconv_block(self, in_channels, out_channels, dropout=False):
        """Upsampling block for decoder"""
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)   # 256 → 128
        e2 = self.enc2(e1)  # 128 → 64
        e3 = self.enc3(e2)  # 64 → 32
        e4 = self.enc4(e3)  # 32 → 16
        e5 = self.enc5(e4)  # 16 → 8
        e6 = self.enc6(e5)  # 8 → 4
        e7 = self.enc7(e6)  # 4 → 2
        e8 = self.enc8(e7)  # 2 → 1
        
        # Decoder with skip connections
        d8 = self.dec8(e8)                       # 1 → 2
        d8 = torch.cat([d8, e7], dim=1)           # Skip connection
        
        d7 = self.dec7(d8)                        # 2 → 4
        d7 = torch.cat([d7, e6], dim=1)           # Skip connection
        
        d6 = self.dec6(d7)                        # 4 → 8
        d6 = torch.cat([d6, e5], dim=1)           # Skip connection
        
        d5 = self.dec5(d6)                        # 8 → 16
        d5 = torch.cat([d5, e4], dim=1)           # Skip connection
        
        d4 = self.dec4(d5)                        # 16 → 32
        d4 = torch.cat([d4, e3], dim=1)           # Skip connection
        
        d3 = self.dec3(d4)                        # 32 → 64
        d3 = torch.cat([d3, e2], dim=1)           # Skip connection
        
        d2 = self.dec2(d3)                        # 64 → 128
        d2 = torch.cat([d2, e1], dim=1)           # Skip connection
        
        d1 = self.dec1(d2)                        # 128 → 256
        
        return d1


# ==================== Discriminator ====================
class Discriminator(nn.Module):
    """PatchGAN discriminator"""
    def __init__(self, in_channels=6):  # Input: [content, sketch] concatenated
        super().__init__()
        
        self.model = nn.Sequential(
            # Layer 1: 256x256 -> 128x128
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: 128x128 -> 64x64
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: 64x64 -> 32x32
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: 32x32 -> 16x16
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 5: 16x16 -> 15x15 (Patch output)
            nn.Conv2d(512, 1, 4, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)


# ==================== GAN Loss with Label Smoothing ====================
class GANLoss(nn.Module):
    """GAN loss with label smoothing for more stable training"""
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
    
    def forward(self, pred, target_is_real):
        # Use label smoothing to prevent discriminator from becoming too strong
        if target_is_real:
            target = torch.ones_like(pred) * 0.9  # Real label smoothed to 0.9
        else:
            target = torch.zeros_like(pred) * 0.1  # Fake label smoothed to 0.1
        return self.loss(pred, target)


# ==================== Main Training Function ====================
def train_pix2pix():
    print("=" * 60)
    print("🎨 PIX2PIX SKETCH TRAINING (STABLE VERSION)")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️ Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # ========== Path configuration ==========
    DATA_ROOT = "/content/drive/MyDrive/style-transfer-project/datasets"
    OUTPUT_DIR = "/content/drive/MyDrive/style-transfer-project/ai/outputs/pix2pix_sketch_stable"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    # ========== Training parameters ==========
    BATCH_SIZE = 4
    EPOCHS = 200
    LR_G = 1e-4      # Generator learning rate
    LR_D = 5e-5      # Discriminator learning rate (lower to prevent overpowering)
    LAMBDA_L1 = 10   # L1 loss weight (reduced from 100)
    IMAGE_SIZE = 256
    
    print(f"\n📊 Training parameters:")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Generator LR: {LR_G}")
    print(f"   Discriminator LR: {LR_D}")
    print(f"   L1 Loss Weight: {LAMBDA_L1}")
    
    # ========== Load dataset ==========
    print("\n📂 Loading dataset...")
    train_dataset = PairedSketchDataset(DATA_ROOT, 'train', IMAGE_SIZE)
    val_dataset = PairedSketchDataset(DATA_ROOT, 'val', IMAGE_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"✅ Training set: {len(train_dataset)} pairs")
    print(f"✅ Validation set: {len(val_dataset)} pairs")
    
    # ========== Create models ==========
    print("\n🤖 Creating models...")
    generator = UNet(in_channels=3, out_channels=3).to(device)
    discriminator = Discriminator(in_channels=6).to(device)
    
    # Try to load pre-trained U-Net from previous training
    best_unet_path = "/content/drive/MyDrive/style-transfer-project/ai/outputs/unet_sketch/best_model.pth"
    if os.path.exists(best_unet_path):
        generator.load_state_dict(torch.load(best_unet_path, map_location=device))
        print(f"✅ Loaded pre-trained U-Net from previous training!")
    else:
        print("⚠️ No pre-trained U-Net found, starting from scratch")
    
    print(f"   Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"   Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # ========== Optimizers ==========
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR_G, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR_D, betas=(0.5, 0.999))
    
    # ========== Loss functions ==========
    gan_loss = GANLoss().to(device)
    l1_loss = nn.L1Loss()
    
    # ========== Learning rate schedulers ==========
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=50, gamma=0.5)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=50, gamma=0.5)
    
    # ========== Early stopping parameters ==========
    patience = 20  # Stop if no improvement for 20 epochs
    patience_counter = 0
    best_val_loss = float('inf')
    
    # ========== Training loop ==========
    print(f"\n🚀 Starting training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        generator.train()
        discriminator.train()
        
        total_G_loss = 0
        total_D_loss = 0
        total_G_gan_loss = 0
        total_G_l1_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for content, sketch in pbar:
            content = content.to(device)
            sketch = sketch.to(device)
            
            # ========== Train Discriminator ==========
            optimizer_D.zero_grad()
            
            # Real images
            real_input = torch.cat([content, sketch], dim=1)
            real_pred = discriminator(real_input)
            loss_D_real = gan_loss(real_pred, True)
            
            # Fake images
            with torch.no_grad():
                fake_sketch = generator(content)
            fake_input = torch.cat([content, fake_sketch], dim=1)
            fake_pred = discriminator(fake_input)
            loss_D_fake = gan_loss(fake_pred, False)
            
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            
            # Gradient clipping for discriminator
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            
            optimizer_D.step()
            
            # ========== Train Generator ==========
            optimizer_G.zero_grad()
            
            # Generate fake sketch
            fake_sketch = generator(content)
            
            # GAN loss
            fake_input = torch.cat([content, fake_sketch], dim=1)
            fake_pred = discriminator(fake_input)
            loss_G_gan = gan_loss(fake_pred, True)
            
            # L1 loss
            loss_G_l1 = l1_loss(fake_sketch, sketch) * LAMBDA_L1
            
            # Total generator loss
            loss_G = loss_G_gan + loss_G_l1
            
            loss_G.backward()
            
            # Gradient clipping for generator
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            
            optimizer_G.step()
            
            # Record losses
            total_G_loss += loss_G.item()
            total_D_loss += loss_D.item()
            total_G_gan_loss += loss_G_gan.item()
            total_G_l1_loss += loss_G_l1.item()
            
            pbar.set_postfix({
                'G': f'{loss_G.item():.2f}',
                'D': f'{loss_D.item():.2f}'
            })
        
        # Update schedulers
        scheduler_G.step()
        scheduler_D.step()
        
        # Calculate average losses
        avg_G_loss = total_G_loss / len(train_loader)
        avg_D_loss = total_D_loss / len(train_loader)
        avg_G_gan = total_G_gan_loss / len(train_loader)
        avg_G_l1 = total_G_l1_loss / len(train_loader)
        
        # Validation
        generator.eval()
        val_loss = 0
        with torch.no_grad():
            for content, sketch in val_loader:
                content = content.to(device)
                sketch = sketch.to(device)
                output = generator(content)
                val_loss += l1_loss(output, sketch).item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Save best generator and early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(generator.state_dict(), os.path.join(OUTPUT_DIR, 'best_generator.pth'))
            print(f"\n💾 Saved best generator (val_loss: {avg_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1}:")
        print(f"   G Loss: {avg_G_loss:.2f} (GAN: {avg_G_gan:.2f}, L1: {avg_G_l1:.2f})")
        print(f"   D Loss: {avg_D_loss:.2f}")
        print(f"   Val Loss: {avg_val_loss:.4f} (Best: {best_val_loss:.4f})")
        print(f"   LR: {optimizer_G.param_groups[0]['lr']:.2e}")
        
        # Visualize every 10 epochs
        if (epoch + 1) % 10 == 0:
            generator.eval()
            with torch.no_grad():
                content_val, sketch_val = next(iter(val_loader))
                content_val = content_val[:4].to(device)
                sketch_val = sketch_val[:4].to(device)
                output_val = generator(content_val)
                
                def denorm(t):
                    return (t.cpu().numpy().transpose(1, 2, 0) + 1) / 2
                
                fig, axes = plt.subplots(4, 3, figsize=(12, 14))
                for i in range(4):
                    # Content
                    axes[i, 0].imshow(np.clip(denorm(content_val[i]), 0, 1))
                    axes[i, 0].set_title('Content')
                    axes[i, 0].axis('off')
                    
                    # Generated
                    axes[i, 1].imshow(np.clip(denorm(output_val[i]), 0, 1))
                    axes[i, 1].set_title('Generated Sketch')
                    axes[i, 1].axis('off')
                    
                    # Target
                    axes[i, 2].imshow(np.clip(denorm(sketch_val[i]), 0, 1))
                    axes[i, 2].set_title('Target Sketch')
                    axes[i, 2].axis('off')
                
                plt.suptitle(f'Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}')
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, f'result_epoch_{epoch+1}.png'))
                plt.show()
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'best_val_loss': best_val_loss,
            }, os.path.join(OUTPUT_DIR, f'checkpoint_epoch_{epoch+1}.pth'))
            print(f"💾 Saved checkpoint at epoch {epoch+1}")
    
    # ========== Save final models ==========
    torch.save(generator.state_dict(), os.path.join(OUTPUT_DIR, 'final_generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(OUTPUT_DIR, 'final_discriminator.pth'))
    
    # Plot training curve
    print(f"\n✅ Training completed!")
    print(f"💾 Best generator: {os.path.join(OUTPUT_DIR, 'best_generator.pth')}")
    print(f"💾 Final generator: {os.path.join(OUTPUT_DIR, 'final_generator.pth')}")
    print("=" * 60)


# ==================== Run Training ====================
if __name__ == "__main__":
    train_pix2pix()