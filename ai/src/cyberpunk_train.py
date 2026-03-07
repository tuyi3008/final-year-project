import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path
import random
import os
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# -------------------------------
# 1. Configuration (Phase 3: Overexposure Fix + Detail Preservation)
# -------------------------------
BATCH_SIZE = 8
IMG_SIZE = 256
EPOCHS = 50  # Total epochs
START_EPOCH = 31  # Start from epoch 31
SAVE_INTERVAL = 2
COCO_NUM = 5000
TOP_K = 200
PATIENCE = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "/kaggle/working/output_top200_phase3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Phase 3: Overexposure fix configuration
CONTENT_WEIGHT = 2.5      # Maintain content weight
STYLE_WEIGHT = 17000.0    # Maintain style strength
EDGE_WEIGHT = 8.0         # Slightly increase edge weight to preserve contours
TV_WEIGHT = 0.001         # Increase TV loss to reduce overexposure (from 0.0005 to 0.001)
COLOR_WEIGHT = 4000.0     # Increase color loss to suppress overexposure (from 2500 to 4000)
LEARNING_RATE = 2e-6      # Keep low learning rate for stable training

writer = SummaryWriter(log_dir=os.path.join(OUTPUT_DIR, "tb"))

# -------------------------------
# 2. Dataset Loading
# -------------------------------
class ImageDataset(Dataset):
    def __init__(self, paths, is_style=False):
        self.paths = paths
        if is_style:
            self.transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(10),
                transforms.ColorJitter(
                    brightness=0.2, 
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05
                ),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except:
            img = Image.open(self.paths[0]).convert("RGB")
        return self.transform(img) * 2 - 1

# Load COCO content images
print("Loading COCO dataset...")
coco_base = Path("/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017")
coco_paths = list(coco_base.rglob("*.jpg"))
random.shuffle(coco_paths)
coco_paths = coco_paths[:COCO_NUM]
print(f"COCO images: {len(coco_paths)}")

# Load Top 200 style images
print("\n=== Loading Top 200 style images ===")
cyber_base = Path("/kaggle/input/datasets/cyanex1702/cyberversecyberpunk-imagesdataset")
all_style_paths = list(cyber_base.rglob("*.jpg"))
top_style_paths = all_style_paths[:200]
print(f"Using {len(top_style_paths)} style images")

# Create datasets
content_dataset = ImageDataset(coco_paths, is_style=False)
style_dataset = ImageDataset(top_style_paths, is_style=True)

content_loader = DataLoader(content_dataset, batch_size=BATCH_SIZE, 
                           shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
style_loader = DataLoader(style_dataset, batch_size=BATCH_SIZE, 
                         shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

print(f"✅ Content DataLoader: {len(content_loader)} batches per epoch")
print(f"✅ Style DataLoader: {len(style_loader)} batches per epoch")

# -------------------------------
# 3. TransformerNet
# -------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels, affine=True)
        )

    def forward(self, x):
        return x + self.block(x)

class TransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 9, 1, 4)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        self.res_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(5)])
        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.conv_out = nn.Conv2d(32, 3, 9, 1, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res_blocks(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.conv_out(y)
        return torch.tanh(y)

# -------------------------------
# 4. VGG16 for Perceptual Loss
# -------------------------------
class VGG16Features(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:23]
        for param in vgg.parameters():
            param.requires_grad = False
        
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg[x])
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, x):
        x = (x + 1) / 2
        x = (x - self.mean) / self.std
        
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        
        return [h1, h2, h3, h4]

def gram_matrix(features):
    """Compute Gram matrix"""
    b, c, h, w = features.size()
    features = features.view(b, c, h * w)
    features = features - features.mean(dim=2, keepdim=True)
    features = F.normalize(features, p=2, dim=2)
    gram = torch.bmm(features, features.transpose(1, 2))
    gram = torch.clamp(gram, -5, 5)
    return gram

# -------------------------------
# 5. Edge Detection Function
# -------------------------------
def edge_detection(x):
    """Sobel edge detection"""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype).to(x.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype).to(x.device)
    
    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
    
    x_gray = x.mean(dim=1, keepdim=True)
    
    edges_x = F.conv2d(x_gray, sobel_x, padding=1)
    edges_y = F.conv2d(x_gray, sobel_y, padding=1)
    edges = torch.sqrt(edges_x**2 + edges_y**2 + 1e-6)
    
    return edges

# -------------------------------
# 6. Adaptive Color Loss (Fix Overexposure)
# -------------------------------
def adaptive_color_loss(output, target):
    """Adaptive color loss, focusing on overexposed areas"""
    output_rgb = (output + 1) / 2
    target_rgb = (target + 1) / 2
    
    # Find overexposed areas (values > 0.85)
    overexposed_mask = (output_rgb > 0.85).float()
    
    loss = 0
    for i in range(3):
        # Apply stronger penalty to overexposed areas
        if overexposed_mask[:, i:i+1, :, :].sum() > 0:
            overexposed_loss = (output_rgb[:, i:i+1, :, :] * overexposed_mask[:, i:i+1, :, :]).mean() * 200
        else:
            overexposed_loss = 0
        
        # Keep original loss for normal areas
        normal_loss = F.mse_loss(
            output_rgb[:, i].mean(), 
            target_rgb[:, i].mean()
        ) * 30
        
        # Variance matching
        var_loss = F.mse_loss(
            output_rgb[:, i].std(), 
            target_rgb[:, i].std()
        ) * 20
        
        loss += overexposed_loss + normal_loss + var_loss
    
    return loss

# -------------------------------
# 7. Detail Preservation Loss
# -------------------------------
def detail_loss(output, content):
    """Preserve high-frequency details"""
    # Laplacian operator to extract high frequencies
    laplacian = torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], 
                              dtype=output.dtype).to(output.device)
    
    # Extract high frequencies from grayscale
    output_gray = output.mean(dim=1, keepdim=True)
    content_gray = content.mean(dim=1, keepdim=True)
    
    output_high = F.conv2d(output_gray, laplacian, padding=1)
    content_high = F.conv2d(content_gray, laplacian, padding=1)
    
    return F.mse_loss(output_high, content_high)

# -------------------------------
# 8. Exposure Control Loss
# -------------------------------
def exposure_loss(output):
    """Prevent overexposure loss"""
    # Limit output to reasonable range [-0.9, 0.9]
    over_exposed = torch.clamp(output - 0.85, min=0)  # Penalize values above 0.85
    under_exposed = torch.clamp(-0.85 - output, min=0)  # Penalize values below -0.85
    return (over_exposed.mean() + under_exposed.mean()) * 500

# -------------------------------
# 9. Initialize Models
# -------------------------------
model = TransformerNet().to(DEVICE)
vgg = VGG16Features().to(DEVICE)
vgg.eval()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                 factor=0.5, patience=8)

# -------------------------------
# 10. Load Best Model from Epoch 30
# -------------------------------
print("\n" + "="*60)
print("🚀 Phase 3: Overexposure Fix + Detail Preservation (Starting from Epoch 31)")
print("="*60)

# Try to load checkpoint from Epoch 30
checkpoint_path = "/kaggle/working/output_top200_phase2_5/checkpoint_epoch_30.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"✓ Loaded stable model from Epoch 30 (Loss: {checkpoint['loss']:.1f})")
else:
    print("⚠️ Checkpoint for Epoch 30 not found, trying to load best_model")
    best_path = "/kaggle/working/output_top200_phase2_5/best_model.pth"
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path))
        print("✓ Loaded best_model")

print(f"\n📊 Phase 3 Configuration (Overexposure Fix):")
print(f"CONTENT_WEIGHT = {CONTENT_WEIGHT}")
print(f"STYLE_WEIGHT = {STYLE_WEIGHT}")
print(f"COLOR_WEIGHT = {COLOR_WEIGHT} (increased)")
print(f"TV_WEIGHT = {TV_WEIGHT} (increased)")
print(f"EDGE_WEIGHT = {EDGE_WEIGHT} (increased)")
print(f"LEARNING_RATE = {LEARNING_RATE}")
print("New: Adaptive color loss, detail loss, exposure control")

# -------------------------------
# 11. Mixed Precision Training
# -------------------------------
scaler = torch.cuda.amp.GradScaler()

# -------------------------------
# 12. Training Loop (Phase 3)
# -------------------------------
best_loss = float('inf')
no_improve_epochs = 0
global_step = 30000  # Starting step from Epoch 30

# Weights for new losses
DETAIL_WEIGHT = 2000.0
EXPOSURE_WEIGHT = 1000.0

print("\n=== Starting Phase 3 Training (Overexposure Fix + Detail Preservation) ===")
for epoch in range(START_EPOCH, EPOCHS+1):
    model.train()
    total_content_loss = 0
    total_style_loss = 0
    total_tv_loss = 0
    total_edge_loss = 0
    total_color_loss = 0
    total_detail_loss = 0
    total_exposure_loss = 0
    valid_batches = 0
    
    style_iter = iter(style_loader)
    
    progress_bar = tqdm(content_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    
    for content_batch in progress_bar:
        content_batch = content_batch.to(DEVICE, non_blocking=True)
        
        try:
            style_batch = next(style_iter)
        except StopIteration:
            style_iter = iter(style_loader)
            style_batch = next(style_iter)
        style_batch = style_batch.to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast():
            output = model(content_batch)
            
            if torch.isnan(output).any():
                continue
            
            content_features = vgg(content_batch)
            output_features = vgg(output)
            style_features = vgg(style_batch)
            
            # Content loss
            content_loss = CONTENT_WEIGHT * (
                F.mse_loss(output_features[1], content_features[1]) * 0.3 +
                F.mse_loss(output_features[2], content_features[2]) * 1.0 +
                F.mse_loss(output_features[3], content_features[3]) * 0.3
            )
            
            # Edge loss
            content_edges = edge_detection(content_batch)
            output_edges = edge_detection(output)
            edge_loss = F.mse_loss(output_edges, content_edges) * EDGE_WEIGHT
            
            # Style loss
            style_loss = 0
            layer_weights = [0.3, 0.7, 1.0, 0.7]
            for i, (out_f, style_f) in enumerate(zip(output_features, style_features)):
                gram_out = gram_matrix(out_f)
                gram_style = gram_matrix(style_f)
                style_loss += F.mse_loss(gram_out, gram_style) * layer_weights[i]
            style_loss = STYLE_WEIGHT * style_loss
            
            # Adaptive color loss (fix overexposure)
            color_loss_value = adaptive_color_loss(output, style_batch) * COLOR_WEIGHT
            
            # Detail preservation loss
            detail_loss_value = detail_loss(output, content_batch) * DETAIL_WEIGHT
            
            # Exposure control loss
            exposure_loss_value = exposure_loss(output) * EXPOSURE_WEIGHT
            
            # TV loss (increased weight to suppress overexposure)
            tv_loss = TV_WEIGHT * (torch.sum(torch.abs(output[:, :, :, :-1] - output[:, :, :, 1:])) + 
                                   torch.sum(torch.abs(output[:, :, :-1, :] - output[:, :, 1:, :])))
            
            total_loss = (content_loss + style_loss + tv_loss + edge_loss + 
                         color_loss_value + detail_loss_value + exposure_loss_value)
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                continue
        
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()
        
        total_content_loss += content_loss.item()
        total_style_loss += style_loss.item()
        total_tv_loss += tv_loss.item()
        total_edge_loss += edge_loss.item()
        total_color_loss += color_loss_value.item()
        total_detail_loss += detail_loss_value.item()
        total_exposure_loss += exposure_loss_value.item()
        valid_batches += 1
        global_step += 1
        
        if global_step % 200 == 0:
            raw_style_value = style_loss.item() / STYLE_WEIGHT
            print(f"\nStep {global_step}: Raw S={raw_style_value:.4f}, "
                  f"Color={color_loss_value.item():.1f}, "
                  f"Detail={detail_loss_value.item():.1f}")
        
        progress_bar.set_postfix({
            'C': f'{content_loss.item():.1f}',
            'S': f'{style_loss.item():.1f}',
            'Col': f'{color_loss_value.item():.1f}',
            'Det': f'{detail_loss_value.item():.1f}',
            'Exp': f'{exposure_loss_value.item():.1f}'
        })
    
    if valid_batches > 0:
        avg_content_loss = total_content_loss / valid_batches
        avg_style_loss = total_style_loss / valid_batches
        avg_tv_loss = total_tv_loss / valid_batches
        avg_edge_loss = total_edge_loss / valid_batches
        avg_color_loss = total_color_loss / valid_batches
        avg_detail_loss = total_detail_loss / valid_batches
        avg_exposure_loss = total_exposure_loss / valid_batches
        avg_total_loss = (avg_content_loss + avg_style_loss + avg_tv_loss + 
                         avg_edge_loss + avg_color_loss + avg_detail_loss + avg_exposure_loss)
    else:
        print("No valid batches, stopping")
        break
    
    writer.add_scalar('Loss/total', avg_total_loss, epoch)
    writer.add_scalar('Loss/content', avg_content_loss, epoch)
    writer.add_scalar('Loss/style', avg_style_loss, epoch)
    writer.add_scalar('Loss/color', avg_color_loss, epoch)
    writer.add_scalar('Loss/detail', avg_detail_loss, epoch)
    writer.add_scalar('Loss/exposure', avg_exposure_loss, epoch)
    writer.add_scalar('Loss/edge', avg_edge_loss, epoch)
    
    raw_style_avg = avg_style_loss / STYLE_WEIGHT
    print(f"\nEpoch {epoch}: Total={avg_total_loss:.1f} | "
          f"C={avg_content_loss:.1f} | S={avg_style_loss:.1f} | "
          f"Raw S={raw_style_avg:.4f} | Col={avg_color_loss:.1f} | "
          f"Det={avg_detail_loss:.1f} | Exp={avg_exposure_loss:.1f} | "
          f"E={avg_edge_loss:.1f} | TV={avg_tv_loss:.1f}")
    
    scheduler.step(avg_total_loss)
    
    if avg_total_loss < best_loss:
        best_loss = avg_total_loss
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
        no_improve_epochs = 0
        print(f"✓ New best model saved!")
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break
    
    if epoch % SAVE_INTERVAL == 0:
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            test_samples = content_batch[:4]
            output_samples = model(test_samples)
            
            comparison = torch.cat([test_samples, output_samples], dim=0)
            save_image(comparison * 0.5 + 0.5, 
                      os.path.join(OUTPUT_DIR, f"epoch_{epoch}_comparison.png"),
                      nrow=4)
            
            save_image(output_samples * 0.5 + 0.5,
                      os.path.join(OUTPUT_DIR, f"epoch_{epoch}_stylized.png"),
                      nrow=2)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_total_loss,
            }, os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch}.pth"))
            
            print(f"✓ Saved samples at epoch {epoch}")
        
        model.train()

print("\n✅ Phase 3 training complete!")
print(f"Best loss: {best_loss:.2f}")
writer.close()