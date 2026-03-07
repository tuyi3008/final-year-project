import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# ==================== TransformerNet (exactly matching training code) ====================
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
        print(f"\n  [Debug] Input shape: {x.shape}, range: [{x.min():.3f}, {x.max():.3f}], mean: {x.mean():.3f}")
        
        y = self.relu(self.in1(self.conv1(x)))
        print(f"  [Debug] After conv1: shape {y.shape}, range [{y.min():.3f}, {y.max():.3f}], mean: {y.mean():.3f}")
        
        y = self.relu(self.in2(self.conv2(y)))
        print(f"  [Debug] After conv2: shape {y.shape}, range [{y.min():.3f}, {y.max():.3f}], mean: {y.mean():.3f}")
        
        y = self.relu(self.in3(self.conv3(y)))
        print(f"  [Debug] After conv3: shape {y.shape}, range [{y.min():.3f}, {y.max():.3f}], mean: {y.mean():.3f}")
        
        y = self.res_blocks(y)
        print(f"  [Debug] After res_blocks: shape {y.shape}, range [{y.min():.3f}, {y.max():.3f}], mean: {y.mean():.3f}")
        
        y = self.relu(self.in4(self.deconv1(y)))
        print(f"  [Debug] After deconv1: shape {y.shape}, range [{y.min():.3f}, {y.max():.3f}], mean: {y.mean():.3f}")
        
        y = self.relu(self.in5(self.deconv2(y)))
        print(f"  [Debug] After deconv2: shape {y.shape}, range [{y.min():.3f}, {y.max():.3f}], mean: {y.mean():.3f}")
        
        y = self.conv_out(y)
        print(f"  [Debug] Output: shape {y.shape}, range [{y.min():.3f}, {y.max():.3f}], mean: {y.mean():.3f}")
        
        return torch.tanh(y)


def test_cyberpunk_model(image_path, model_path, output_path=None, show_result=True):
    """
    Test Cyberpunk model function with detailed debug information
    """
    print("="*60)
    print("🎨 Cyberpunk Model Test")
    print("="*60)
    
    # Check if files exist
    print(f"\n[1] Checking files...")
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    print(f"✅ Image: {image_path}")
    print(f"✅ Model: {model_path}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[2] Using device: {device}")
    
    # Load model
    print(f"\n[3] Loading model...")
    model = TransformerNet().to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        print(f"✅ Checkpoint loaded, type: {type(checkpoint)}")
        
        # Analyze checkpoint structure
        if isinstance(checkpoint, dict):
            print(f"   Checkpoint keys: {list(checkpoint.keys())}")
            
            if 'model_state_dict' in checkpoint:
                # Try to load model_state_dict
                model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Loaded model_state_dict")
                
                # Print loss and epoch info if available
                if 'loss' in checkpoint:
                    print(f"   Model loss: {checkpoint['loss']:.2f}")
                if 'epoch' in checkpoint:
                    print(f"   Model epoch: {checkpoint['epoch']}")
                    
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print("✅ Loaded state_dict")
            else:
                # Try direct loading
                try:
                    model.load_state_dict(checkpoint)
                    print("✅ Loaded direct state_dict")
                except Exception as e:
                    print(f"   ⚠️ Direct loading failed: {e}")
                    # If failed, try filtered loading
                    print("   Trying filtered loading...")
                    model_dict = model.state_dict()
                    filtered_dict = {k: v for k, v in checkpoint.items() 
                                   if k in model_dict and v.shape == model_dict[k].shape}
                    model.load_state_dict(filtered_dict, strict=False)
                    print(f"   Loaded {len(filtered_dict)}/{len(model_dict)} layers")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Loaded direct state_dict")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    model.eval()
    
    # Load and preprocess image
    print(f"\n[4] Loading and preprocessing image...")
    try:
        # Original image
        img = Image.open(image_path).convert('RGB')
        print(f"   Original image size: {img.size}")
        
        # Show original image statistics
        img_np = np.array(img)
        print(f"   Original image stats - min: {img_np.min()}, max: {img_np.max()}, mean: {img_np.mean():.1f}")
        
        # Define transform (exactly matching training)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(device)
        print(f"   Tensor shape: {img_tensor.shape}")
        print(f"   Tensor range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}], mean: {img_tensor.mean():.3f}")
        
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None
    
    # Model inference
    print(f"\n[5] Running inference...")
    try:
        with torch.no_grad():
            output = model(img_tensor)
            
        print(f"\n[6] Raw output stats:")
        print(f"   Shape: {output.shape}")
        print(f"   Range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"   Mean: {output.mean():.3f}")
        print(f"   Std: {output.std():.3f}")
        
        # Analyze output distribution
        output_np = output[0].cpu().numpy()
        print(f"\n[7] Output channel stats:")
        for c in range(3):
            channel = output_np[c]
            print(f"   Channel {c}: min={channel.min():.3f}, max={channel.max():.3f}, mean={channel.mean():.3f}")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return None
    
    # Post-processing: tanh output range is [-1, 1], need to convert to [0, 1]
    print(f"\n[8] Post-processing (tanh -> [0,1])...")
    
    def postprocess_tanh(tensor):
        """Process tanh output [-1, 1] -> [0, 1]"""
        result = (tensor + 1) / 2
        return np.clip(result, 0, 1)
    
    def postprocess_clamp(tensor):
        """Direct clamp to [0,1]"""
        return np.clip(tensor, 0, 1)
    
    def postprocess_norm(tensor):
        """Normalize to [0,1]"""
        t_min = tensor.min()
        t_max = tensor.max()
        return (tensor - t_min) / (t_max - t_min + 1e-8)
    
    # Test different post-processing methods
    methods = [
        ("Tanh -> [0,1] (recommended)", postprocess_tanh),
        ("Direct clamp", postprocess_clamp),
        ("Min-max norm", postprocess_norm),
    ]
    
    results = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    for i, (name, method) in enumerate(methods):
        processed = method(output_np)
        processed = np.transpose(processed, (1, 2, 0))  # CHW -> HWC
        processed = np.clip(processed, 0, 1)
        
        print(f"\n   {name}:")
        print(f"      Range: [{processed.min():.3f}, {processed.max():.3f}]")
        print(f"      Mean: {processed.mean():.3f}")
        results.append((name, processed))
        
        # Display results
        axes[(i+1)//2, (i+1)%2].imshow(processed)
        axes[(i+1)//2, (i+1)%2].set_title(name)
        axes[(i+1)//2, (i+1)%2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Save result (using tanh method)
    if output_path:
        print(f"\n[9] Saving result...")
        try:
            result_to_save = postprocess_tanh(output_np)
            result_to_save = np.transpose(result_to_save, (1, 2, 0))
            result_to_save = (result_to_save * 255).astype(np.uint8)
            Image.fromarray(result_to_save).save(output_path)
            print(f"✅ Result saved to: {output_path}")
        except Exception as e:
            print(f"❌ Error saving result: {e}")
    
    print("\n" + "="*60)
    print("✅ Test complete!")
    print("="*60)
    
    return results


def analyze_model_weights(model_path):
    """
    Analyze model weights statistics
    """
    print("="*60)
    print("📊 Model Weight Analysis")
    print("="*60)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    print(f"\nModel weights statistics:")
    total_params = 0
    for name, param in state_dict.items():
        if 'weight' in name or 'bias' in name:
            print(f"\n  {name}:")
            print(f"    Shape: {param.shape}")
            print(f"    Range: [{param.min():.3f}, {param.max():.3f}]")
            print(f"    Mean: {param.mean():.3f}")
            print(f"    Std: {param.std():.3f}")
            total_params += param.numel()
    
    print(f"\nTotal parameters: {total_params:,}")


if __name__ == "__main__":
    # ==================== Configure paths ====================
    MODEL_PATH = r"D:\Final_Year_Project\final-year-project\models\cyberpunk.pt"
    TEST_IMAGE = r"D:\Final_Year_Project\final-year-project\ai\test1.jpg"
    OUTPUT_IMAGE = r"D:\Final_Year_Project\final-year-project\ai\outputs\test_cyberpunk.jpg"
    
    # ==================== Run tests ====================
    
    # 1. First analyze model weights
    print("\n" + "="*60)
    print("STEP 1: Analyzing model weights")
    print("="*60)
    analyze_model_weights(MODEL_PATH)
    
    # 2. Test single image
    print("\n" + "="*60)
    print("STEP 2: Testing single image")
    print("="*60)
    test_cyberpunk_model(TEST_IMAGE, MODEL_PATH, OUTPUT_IMAGE, show_result=True)