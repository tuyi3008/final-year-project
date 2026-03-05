import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os

class ConvNormLReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, pad_mode="reflect", groups=1, bias=False):
        pad_layer = {
            "zero": nn.ZeroPad2d,
            "same": nn.ReplicationPad2d,
            "reflect": nn.ReflectionPad2d,
        }
        super(ConvNormLReLU, self).__init__(
            pad_layer[pad_mode](padding),
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=bias),
            nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

class InvertedResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion_ratio=2):
        super(InvertedResBlock, self).__init__()
        self.use_res_connect = in_ch == out_ch
        bottleneck = int(round(in_ch*expansion_ratio))
        layers = []
        if expansion_ratio != 1:
            layers.append(ConvNormLReLU(in_ch, bottleneck, kernel_size=1, padding=0))
        
        layers.append(ConvNormLReLU(bottleneck, bottleneck, groups=bottleneck, bias=True))
        layers.append(nn.Conv2d(bottleneck, out_ch, kernel_size=1, padding=0, bias=False))
        layers.append(nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, input):
        out = self.layers(input)
        if self.use_res_connect:
            out = input + out
        return out

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.block_a = nn.Sequential(
            ConvNormLReLU(3,  32, kernel_size=7, padding=3),
            ConvNormLReLU(32, 64, stride=2, padding=(0,1,0,1)),
            ConvNormLReLU(64, 64)
        )
        
        self.block_b = nn.Sequential(
            ConvNormLReLU(64,  128, stride=2, padding=(0,1,0,1)),            
            ConvNormLReLU(128, 128)
        )
        
        self.block_c = nn.Sequential(
            ConvNormLReLU(128, 128),
            InvertedResBlock(128, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            ConvNormLReLU(256, 128),
        )    
        
        self.block_d = nn.Sequential(
            ConvNormLReLU(128, 128),
            ConvNormLReLU(128, 128)
        )

        self.block_e = nn.Sequential(
            ConvNormLReLU(128, 64),
            ConvNormLReLU(64,  64),
            ConvNormLReLU(64,  32, kernel_size=7, padding=3)
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )
        
    def forward(self, input, align_corners=True):
        out = self.block_a(input)
        half_size = out.size()[-2:]
        out = self.block_b(out)
        out = self.block_c(out)
        
        if align_corners:
            out = F.interpolate(out, half_size, mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.block_d(out)

        if align_corners:
            out = F.interpolate(out, input.size()[-2:], mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.block_e(out)

        out = self.out_layer(out)
        return out

def test_anime_model(image_path, model_path, model_name, output_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📌 test model: {model_name}")
    
    # load model
    model = Generator().to(device)
    
    try:
        # load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint, strict=False)
            print(f"   ✅ model loadede successfully")
        else:
            model.load_state_dict(checkpoint)
            print(f"   ✅ model loadede successfully")
        
        model.eval()
        
    except Exception as e:
        print(f"   ❌ model loading failed: {e}")
        return None
    
    # preprocess
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # load image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # inference
    with torch.no_grad():
        output = model(img_tensor, align_corners=True)
    
    # postprocess
    output = output.squeeze(0).cpu()
    output = (output * 0.5 + 0.5).clamp(0, 1)
    output = transforms.ToPILImage()(output)
    
    # save output
    if output_path:
        output.save(output_path)
        print(f"   ✅ Output saved: {output_path}")
    
    return img, output

# ==================== Main ====================
if __name__ == "__main__":
    BASE_DIR = r"D:\Final_Year_Project\final-year-project\ai"
    
    MODELS = {
        "Hayao": os.path.join(BASE_DIR, "outputs", "pytorch_generator_Hayao.pt"),
        "Shinkai": os.path.join(BASE_DIR, "outputs", "pytorch_generator_Shinkai.pt"),
        "Paprika": os.path.join(BASE_DIR, "outputs", "pytorch_generator_Paprika.pt")
    }
    
    TEST_IMAGE = os.path.join(BASE_DIR, "test5.jpg")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "anime_results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("="*60)
    
    for name, path in MODELS.items():
        if os.path.exists(path):
            out_path = os.path.join(OUTPUT_DIR, f"{name}_official.jpg")
            test_anime_model(TEST_IMAGE, path, name, out_path)
    
    print(f"\n✅ Results saved in: {OUTPUT_DIR}")