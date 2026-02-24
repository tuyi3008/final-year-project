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
from datasets import load_dataset

# ==================== PhotoSketching Dataset ====================
class PhotoSketchDataset(Dataset):
    """PhotoSketching dataset from Hugging Face"""
    def __init__(self, split='train', img_size=256, transform=None):
        self.img_size = img_size
        
        # 加载 Hugging Face 数据集
        print(f"📂 加载 PhotoSketching {split} 集...")
        ds = load_dataset("rhfeiyang/photo-sketch-pair-500")
        self.dataset = ds[split]
        print(f"✅ 加载{split}集: {len(self.dataset)}对图片")
        
        # Transform
        if transform is None:
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
        else:
            self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        photo = item['photo'].convert('RGB')
        sketch = item['sketch'].convert('RGB')
        
        if self.transform:
            photo = self.transform(photo)
            sketch = self.transform(sketch)
            
        return photo, sketch


# ==================== U-Net Model ====================
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64, batchnorm=False)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 512)
        self.enc6 = self.conv_block(512, 512)
        self.enc7 = self.conv_block(512, 512)
        self.enc8 = self.conv_block(512, 512, batchnorm=False)
        
        # Decoder
        self.dec8 = self.upconv_block(512, 512, dropout=True)
        self.dec7 = self.upconv_block(1024, 512, dropout=True)
        self.dec6 = self.upconv_block(1024, 512, dropout=True)
        self.dec5 = self.upconv_block(1024, 512)
        self.dec4 = self.upconv_block(1024, 256)
        self.dec3 = self.upconv_block(512, 128)
        self.dec2 = self.upconv_block(256, 64)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def conv_block(self, in_channels, out_channels, batchnorm=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)
    
    def upconv_block(self, in_channels, out_channels, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)
        
        d8 = self.dec8(e8)
        d8 = torch.cat([d8, e7], dim=1)
        d7 = self.dec7(d8)
        d7 = torch.cat([d7, e6], dim=1)
        d6 = self.dec6(d7)
        d6 = torch.cat([d6, e5], dim=1)
        d5 = self.dec5(d6)
        d5 = torch.cat([d5, e4], dim=1)
        d4 = self.dec4(d5)
        d4 = torch.cat([d4, e3], dim=1)
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d1 = self.dec1(d2)
        return d1


# ==================== 继续训练函数 ====================
def continue_training():
    print("=" * 60)
    print("🎨 CONTINUE TRAINING U-NET (FROM BEST MODEL)")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️ Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # ========== 路径配置 ==========
    PRETRAINED_PATH = "/kaggle/input/models/tuyi3008/final-sketch-model/pytorch/default/1/final_sketch_model.pth"
    OUTPUT_DIR = "/kaggle/working/unet_photosketch_continued"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ========== 训练参数 ==========
    BATCH_SIZE = 8
    EPOCHS = 50  # 继续训练100轮
    LEARNING_RATE = 5e-5  # 用小一点的学习率微调
    IMAGE_SIZE = 256
    
    print(f"\n📊 Training parameters:")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    
    # ========== 加载数据 ==========
    print("\n📂 Loading dataset...")
    full_dataset = PhotoSketchDataset(split='train', img_size=IMAGE_SIZE)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"✅ Training set: {len(train_dataset)} pairs")
    print(f"✅ Validation set: {len(val_dataset)} pairs")
    
    # ========== 创建模型并加载预训练权重 ==========
    print("\n🤖 Loading pre-trained model...")
    model = UNet().to(device)
    
    if os.path.exists(PRETRAINED_PATH):
        model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=device))
        print(f"✅ Loaded pre-trained model from {PRETRAINED_PATH}")
    else:
        print(f"❌ Pre-trained model not found at {PRETRAINED_PATH}")
        return
    
    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ========== 优化器 ==========
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    l1_loss = nn.L1Loss()
    
    # ========== 学习率调度器 ==========
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # ========== 训练循环 ==========
    print(f"\n🚀 Continuing training for {EPOCHS} epochs...")
    
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for photo, sketch in pbar:
            photo = photo.to(device)
            sketch = sketch.to(device)
            
            optimizer.zero_grad()
            output = model(photo)
            loss = l1_loss(output, sketch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for photo, sketch in val_loader:
                photo = photo.to(device)
                sketch = sketch.to(device)
                output = model(photo)
                val_loss += l1_loss(output, sketch).item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
            print(f"\n💾 Saved best model (val_loss: {avg_val_loss:.4f})")
        
        print(f"\n📊 Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Visualize every 10 epochs
        if (epoch + 1) % 10 == 0:
            # Save checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(OUTPUT_DIR, f'checkpoint_epoch_{epoch+1}.pth'))
            
            # Visualize
            model.eval()
            with torch.no_grad():
                photo_val, sketch_val = next(iter(val_loader))
                photo_val = photo_val[:4].to(device)
                sketch_val = sketch_val[:4].to(device)
                output_val = model(photo_val)
                
                def denorm(t):
                    return (t.cpu().numpy().transpose(1, 2, 0) + 1) / 2
                
                fig, axes = plt.subplots(4, 3, figsize=(12, 14))
                for i in range(4):
                    axes[i, 0].imshow(np.clip(denorm(photo_val[i]), 0, 1))
                    axes[i, 0].set_title('Photo')
                    axes[i, 0].axis('off')
                    
                    axes[i, 1].imshow(np.clip(denorm(output_val[i]), 0, 1))
                    axes[i, 1].set_title('Generated')
                    axes[i, 1].axis('off')
                    
                    axes[i, 2].imshow(np.clip(denorm(sketch_val[i]), 0, 1))
                    axes[i, 2].set_title('Target')
                    axes[i, 2].axis('off')
                
                plt.suptitle(f'Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}')
                plt.tight_layout()
                plt.show()
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'final_model.pth'))
    
    print(f"\n✅ Training completed!")
    print(f"💾 Best model: {os.path.join(OUTPUT_DIR, 'best_model.pth')}")
    print("=" * 60)


if __name__ == "__main__":
    continue_training()