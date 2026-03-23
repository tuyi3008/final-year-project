import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration (Optimized for Kaggle T4/P100) ====================
class Config:
    # dataset paths - Kaggle path
    trainA_dir = '/kaggle/input/datasets/vanmanhbk/ukiyoe2photo-v2-256/ukiyoe2photo-v2-256/trainA'
    trainB_dir = '/kaggle/input/datasets/vanmanhbk/ukiyoe2photo-v2-256/ukiyoe2photo-v2-256/trainB'
    
    # training parameters
    batch_size = 8
    num_epochs = 80
    num_workers = 4
    image_size = 256
    
    # optimizer parameters
    lr = 2e-4
    beta1 = 0.5
    beta2 = 0.999
    
    # loss weights
    lambda_cycle = 10.0
    lambda_identity = 5.0
    
    # learning rate decay
    lr_decay_start_epoch = 40          
    
    # saving parameters
    save_interval = 20                
    
    # training optimizations
    use_amp = True                      # keep mixed precision for speed
    update_D_every = 1                  
    use_checkpointing = False          
    accumulate_grads = 4               
    
    # training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # cuDNN options for reproducibility and speed
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

cfg = Config()

class FastDataset(Dataset):
    def __init__(self, dirA, dirB, transform=None):
        self.transform = transform
        
        self.imagesA = sorted([os.path.join(dirA, f) for f in os.listdir(dirA) 
                               if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.imagesB = sorted([os.path.join(dirB, f) for f in os.listdir(dirB) 
                               if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        print(f"Dataset A: {len(self.imagesA)} images")
        print(f"Dataset B: {len(self.imagesB)} images")
        
        # Load all into memory (limit to 2000 images for speed)
        print("Loading data into memory...")
        self.dataA = []
        for path in tqdm(self.imagesA[:2000], desc="Loading A"):
            img = Image.open(path).convert('RGB')
            if transform:
                img = transform(img)
            self.dataA.append(img)
        
        self.dataB = []
        for path in tqdm(self.imagesB[:2000], desc="Loading B"):
            img = Image.open(path).convert('RGB')
            if transform:
                img = transform(img)
            self.dataB.append(img)
        
        print(f"Loaded {len(self.dataA)} A images, {len(self.dataB)} B images")
        
    def __len__(self):
        return max(len(self.dataA), len(self.dataB))
    
    def __getitem__(self, idx):
        return {
            'A': self.dataA[idx % len(self.dataA)],
            'B': self.dataB[idx % len(self.dataB)]
        }

# ==================== Generator (Lightweight + Spectral Normalization) ====================
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(dim, dim, 3, 1, 0)),  # add spectral norm for stability
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(dim, dim, 3, 1, 0)),
            nn.InstanceNorm2d(dim)
        )
    
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residuals=4):  # 9→4, lightweight
        super().__init__()
        
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7, 1, 0),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        in_dim = 64
        out_dim = 128
        for _ in range(2):
            model += [
                nn.Conv2d(in_dim, out_dim, 3, 2, 1),
                nn.InstanceNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ]
            in_dim = out_dim
            out_dim *= 2
        
        self.res_blocks = nn.ModuleList()
        for _ in range(n_residuals):
            self.res_blocks.append(ResidualBlock(in_dim))
        
        out_dim = in_dim // 2
        for _ in range(2):
            model += [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_dim, out_dim, 3, 1, 1),
                nn.InstanceNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ]
            in_dim = out_dim
            out_dim //= 2
        
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, 7, 1, 0),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

# ==================== Discriminator (Lightweight + Spectral Normalization) ====================
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        def conv_block(in_dim, out_dim, stride=2):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_dim, out_dim, 4, stride, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
        # 4 layers → 3 layers, lightweight
        self.model = nn.Sequential(
            conv_block(in_channels, 64, stride=2),
            conv_block(64, 128, stride=2),
            conv_block(128, 256, stride=2),
            nn.Conv2d(256, 1, 4, 1, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# ==================== Optimized Trainer (Fixed Patch Size Issue) ====================
class CycleGANTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.step = 0
        self.grad_accum_step = 0  # gradient accumulation counter
        
        print(f"Initializing models on {self.device}...")
        
        # Initialize networks
        self.G_A2B = Generator().to(self.device)
        self.G_B2A = Generator().to(self.device)
        self.D_A = Discriminator().to(self.device)
        self.D_B = Discriminator().to(self.device)
        
        self._init_weights()
        
        # Optimizers
        self.optim_G = optim.Adam(
            list(self.G_A2B.parameters()) + list(self.G_B2A.parameters()),
            lr=cfg.lr, betas=(cfg.beta1, cfg.beta2)
        )
        self.optim_D = optim.Adam(
            list(self.D_A.parameters()) + list(self.D_B.parameters()),
            lr=cfg.lr, betas=(cfg.beta1, cfg.beta2)
        )
        
        # Learning rate schedulers (optimized for 80 epochs)
        self.scheduler_G = optim.lr_scheduler.LambdaLR(
            self.optim_G, 
            lr_lambda=lambda epoch: 1.0 if epoch < cfg.lr_decay_start_epoch 
            else max(0.0, 1.0 - (epoch - cfg.lr_decay_start_epoch) / 40)
        )
        self.scheduler_D = optim.lr_scheduler.LambdaLR(
            self.optim_D,
            lr_lambda=lambda epoch: 1.0 if epoch < cfg.lr_decay_start_epoch
            else max(0.0, 1.0 - (epoch - cfg.lr_decay_start_epoch) / 40)
        )
        
        # Loss functions
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
        # Mixed precision
        self.scaler_G = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
        self.scaler_D = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
        
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('samples', exist_ok=True)
        
        print("Initialization complete!")
        
    def _init_weights(self):
        def init(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
        self.G_A2B.apply(init)
        self.G_B2A.apply(init)
        self.D_A.apply(init)
        self.D_B.apply(init)
    
    def train_step(self, real_A, real_B):
        """Single training step - Fixed Patch Size + Gradient Accumulation"""
        batch_size = real_A.size(0)
        
        # ========== 1. Generator Forward ==========
        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            fake_B = self.G_A2B(real_A)
            fake_A = self.G_B2A(real_B)
        
        # ========== 2. Train Generator (with gradient accumulation) ==========
        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            # Identity loss
            id_A = self.G_B2A(real_A)
            id_B = self.G_A2B(real_B)
            loss_id = (self.criterion_identity(id_A, real_A) + 
                      self.criterion_identity(id_B, real_B)) * cfg.lambda_identity
            
            # GAN loss - Key fix: automatically match patch size
            pred_fake_B = self.D_B(fake_B)
            pred_fake_A = self.D_A(fake_A)
            
            # Dynamically generate labels (match discriminator output size)
            real_label = torch.ones_like(pred_fake_B, device=self.device)
            fake_label = torch.zeros_like(pred_fake_B, device=self.device)
            
            loss_gan = self.criterion_GAN(pred_fake_B, real_label) + \
                      self.criterion_GAN(pred_fake_A, real_label)
            
            # Cycle loss
            rec_A = self.G_B2A(fake_B)
            rec_B = self.G_A2B(fake_A)
            loss_cycle = (self.criterion_cycle(rec_A, real_A) + 
                         self.criterion_cycle(rec_B, real_B)) * cfg.lambda_cycle
            
            loss_G = (loss_gan + loss_cycle + loss_id) / cfg.accumulate_grads  # scale loss
        
        self.scaler_G.scale(loss_G).backward()
        self.grad_accum_step += 1
        
        # Update after accumulating specified steps
        if self.grad_accum_step % cfg.accumulate_grads == 0:
            self.scaler_G.step(self.optim_G)
            self.scaler_G.update()
            self.optim_G.zero_grad()
            self.grad_accum_step = 0
        
        # ========== 3. Train Discriminator (update every step) ==========
        self.optim_D.zero_grad()
        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            # D_A
            pred_real_A = self.D_A(real_A)
            pred_fake_A = self.D_A(fake_A.detach())
            
            # Dynamically generate labels for D_A (match output size)
            real_label_A = torch.ones_like(pred_real_A, device=self.device)
            fake_label_A = torch.zeros_like(pred_fake_A, device=self.device)
            
            loss_D_A = (self.criterion_GAN(pred_real_A, real_label_A) + 
                       self.criterion_GAN(pred_fake_A, fake_label_A)) * 0.5
            
            # D_B
            pred_real_B = self.D_B(real_B)
            pred_fake_B = self.D_B(fake_B.detach())
            
            # Dynamically generate labels for D_B (match output size)
            real_label_B = torch.ones_like(pred_real_B, device=self.device)
            fake_label_B = torch.zeros_like(pred_fake_B, device=self.device)
            
            loss_D_B = (self.criterion_GAN(pred_real_B, real_label_B) + 
                       self.criterion_GAN(pred_fake_B, fake_label_B)) * 0.5
            
            loss_D = (loss_D_A + loss_D_B) / cfg.accumulate_grads  # scale loss
        
        self.scaler_D.scale(loss_D).backward()
        if self.grad_accum_step % cfg.accumulate_grads == 0:
            self.scaler_D.step(self.optim_D)
            self.scaler_D.update()
            self.optim_D.zero_grad()
        
        self.step += 1
        
        return {
            'G': loss_G.item() * cfg.accumulate_grads,  # restore actual loss value
            'D': loss_D.item() * cfg.accumulate_grads
        }
    
    def save_samples(self, epoch, dataloader):
        """Reduce save frequency for speed"""
        self.G_A2B.eval()
        self.G_B2A.eval()
        
        with torch.no_grad():
            batch = next(iter(dataloader))
            real_A = batch['A'][:2].to(self.device)  # reduce computation
            real_B = batch['B'][:2].to(self.device)
            
            fake_B = self.G_A2B(real_A)
            fake_A = self.G_B2A(real_B)
            
            comparison = torch.cat([real_A, fake_B, real_B, fake_A], dim=0)
            save_image(comparison, f'samples/epoch_{epoch}.png', nrow=2, normalize=True)
        
        self.G_A2B.train()
        self.G_B2A.train()
    
    def save_checkpoint(self, epoch):
        """Only save generator to reduce I/O"""
        torch.save({
            'epoch': epoch,
            'G_A2B': self.G_A2B.state_dict(),
            'optim_G': self.optim_G.state_dict(),
        }, f'checkpoints/epoch_{epoch}.pth')
        print(f"✓ Saved checkpoint: epoch_{epoch}.pth")
    
    def train(self, train_loader):
        print(f"\nStarting training for {cfg.num_epochs} epochs")
        print(f"Batch size: {cfg.batch_size} (accumulate 4 steps = effective 8)")
        print(f"Iterations per epoch: {len(train_loader)}")
        print("-" * 50)
        
        for epoch in range(1, cfg.num_epochs + 1):
            # Update learning rate
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            current_lr = self.optim_G.param_groups[0]['lr']
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{cfg.num_epochs}')
            epoch_loss_G = 0
            epoch_loss_D = 0
            
            for batch in pbar:
                real_A = batch['A'].to(self.device, non_blocking=True)
                real_B = batch['B'].to(self.device, non_blocking=True)
                
                losses = self.train_step(real_A, real_B)
                
                epoch_loss_G += losses['G']
                epoch_loss_D += losses['D']
                
                # Reduce pbar update frequency for speed
                if self.step % 10 == 0:
                    pbar.set_postfix({
                        'G': f'{losses["G"]:.2f}',
                        'D': f'{losses["D"]:.2f}',
                        'LR': f'{current_lr:.6f}'
                    })
            
            # Only print key info to reduce I/O
            avg_G = epoch_loss_G / len(train_loader)
            avg_D = epoch_loss_D / len(train_loader)
            if epoch % 10 == 0:
                print(f"\nEpoch {epoch}: G={avg_G:.3f}, D={avg_D:.3f}\n")
            
            # Save every 20 epochs to reduce I/O
            if epoch % cfg.save_interval == 0:
                self.save_samples(epoch, train_loader)
                self.save_checkpoint(epoch)
            
            torch.cuda.empty_cache()

# ==================== Main Program ====================
if __name__ == "__main__":
    print("=" * 50)
    print("CycleGAN Training (Kaggle Optimized Version - Fixed Patch Size)")
    print("=" * 50)
    
    # Data preprocessing (use fastest interpolation)
    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size), 
                          interpolation=transforms.InterpolationMode.NEAREST),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    # Create dataset
    print("\nLoading dataset...")
    dataset = FastDataset(cfg.trainA_dir, cfg.trainB_dir, transform=transform)
    train_loader = DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True  # drop incomplete batch for speed
    )
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {cfg.batch_size} (accumulate 4 steps = effective 8)")
    print(f"  Iterations per epoch: {len(train_loader)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if torch.cuda.is_available() else "  CPU")
    
    trainer = CycleGANTrainer(cfg)
    trainer.train(train_loader)