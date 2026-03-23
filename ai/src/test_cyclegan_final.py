import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

class TestConfig:
    model_path = r"D:\Final_Year_Project\final-year-project\models\epoch_25.pth" 
    input_image_path = r"D:\Final_Year_Project\final-year-project\ai\test9.jpg"
    output_image_path = r"D:\Final_Year_Project\final-year-project\ai\test_output_correct.jpg"
    
    image_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    sharpen_strength = 1.8
    contrast_strength = 1.2
    saturation_strength = 1.3
    brightness_strength = 1.1
    edge_enhance_strength = 1.2
    detail_strength = 1.5

cfg = TestConfig()

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim)
        )
    
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, n_residuals=9):
        super().__init__()
        
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        model += [
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        ]
        
        for _ in range(n_residuals):
            model.append(ResidualBlock(256))
        
        model += [
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

def post_process_image_cv2(image_cv2, config):
    img = image_cv2.copy()
    
    # Sharpening
    kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel_sharpen)
    
    # Edge enhancement
    if config.edge_enhance_strength > 1.0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_enhanced = cv2.addWeighted(edges_colored, config.edge_enhance_strength - 1.0, np.zeros_like(edges_colored), 0, 0)
        img = cv2.addWeighted(img, 1.0, edges_enhanced, 0.2, 0)
    
    # Detail enhancement (unsharp mask)
    blurred = cv2.GaussianBlur(img, (0, 0), 3.0)
    img = cv2.addWeighted(img, 1.0 + config.detail_strength, blurred, -config.detail_strength, 0)
    
    # Contrast enhancement
    img = cv2.convertScaleAbs(img, alpha=config.contrast_strength, beta=0)
    
    # Convert to HSV for saturation adjustment
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * config.saturation_strength
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] = hsv[:, :, 2] * config.brightness_strength
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Denoising
    img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
    
    # Final subtle sharpening
    kernel_final = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    img = cv2.filter2D(img, -1, kernel_final)
    
    return img

def test_correct_generator():
    print(f"Using device: {cfg.device}")
    model = Generator().to(cfg.device)
    model.eval()

    print(f"\nLoading model: {cfg.model_path}")
    try:
        checkpoint = torch.load(cfg.model_path, map_location=cfg.device, weights_only=False)
        
        if isinstance(checkpoint, dict):
            print(f"Model keys: {list(checkpoint.keys())}")
            
            if 'G_BA_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['G_BA_state_dict'])
                print("Loaded G_BA (photo -> ukiyo-e) weights")
            elif 'G_AB_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['G_AB_state_dict'])
                print("Loaded G_AB (ukiyoe -> photo) weights")
            else:
                model.load_state_dict(checkpoint)
                print("Loaded direct weights")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded direct state_dict")
    
    except Exception as e:
        print(f"Model loading failed: {str(e)}")
        return

    print(f"\nLoading image: {cfg.input_image_path}")
    if not os.path.exists(cfg.input_image_path):
        print(f"Input image not found: {cfg.input_image_path}")
        return
    
    # Load with PIL for preprocessing
    image_pil = Image.open(cfg.input_image_path).convert('RGB')
    original_size = image_pil.size

    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    input_tensor = transform(image_pil).unsqueeze(0).to(cfg.device)

    print("\nGenerating ukiyo-e style image...")
    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_tensor = (output_tensor + 1) / 2.0
    output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

    output_tensor = output_tensor.squeeze(0).cpu()
    output_np = output_tensor.permute(1, 2, 0).numpy()
    output_np = (output_np * 255.0).astype('uint8')
    
    # Convert RGB to BGR for OpenCV
    output_cv2 = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
    
    # Resize to original size
    output_cv2 = cv2.resize(output_cv2, original_size, interpolation=cv2.INTER_LANCZOS4)
    
    # Also keep RGB version for comparison
    output_rgb = cv2.cvtColor(output_cv2, cv2.COLOR_BGR2RGB)
    output_pil = Image.fromarray(output_rgb)

    print("\nApplying post-processing...")
    output_enhanced_cv2 = post_process_image_cv2(output_cv2, cfg)
    output_enhanced_rgb = cv2.cvtColor(output_enhanced_cv2, cv2.COLOR_BGR2RGB)
    output_enhanced_pil = Image.fromarray(output_enhanced_rgb)

    output_enhanced_pil.save(cfg.output_image_path, quality=95)
    print(f"\nOutput saved: {cfg.output_image_path}")

    # Create comparison image
    original_resized = image_pil.resize(original_size)
    comparison_width = original_size[0] * 3
    comparison_height = original_size[1]
    comparison = Image.new('RGB', (comparison_width, comparison_height))
    
    comparison.paste(original_resized, (0, 0))
    comparison.paste(output_pil, (original_size[0], 0))
    comparison.paste(output_enhanced_pil, (original_size[0] * 2, 0))
    
    comparison_path = cfg.output_image_path.replace('.jpg', '_comparison.jpg')
    comparison.save(comparison_path, quality=95)
    print(f"Comparison image saved: {comparison_path}")

if __name__ == "__main__":
    print("=" * 60)
    print("CycleGAN Ukiyo-e Style Transfer")
    print("=" * 60)
    print(f"Sharpen: {cfg.sharpen_strength}")
    print(f"Contrast: {cfg.contrast_strength}")
    print(f"Saturation: {cfg.saturation_strength}")
    print(f"Edge enhance: {cfg.edge_enhance_strength}")
    print("=" * 60 + "\n")
    
    test_correct_generator()