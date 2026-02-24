import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os

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

class PostProcessModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1),
        )
        
    def forward(self, x):
        residual = self.refine(x)
        return torch.clamp(x + 0.05 * residual, -1, 1)

def test_single_image(image_path, model_path, output_path=None, show_result=True, use_post_process=True):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Use Device: {device}")
    
    model = UNet().to(device)
    post_processor = PostProcessModule().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
 
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'post_processor_state_dict' in checkpoint and use_post_process:
                post_processor.load_state_dict(checkpoint['post_processor_state_dict'])
                print("✅ The complete model (including post-processor) has been loaded.")
            else:
                print("✅ Loaded model weights")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Loaded direct model weights")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Loaded direct model weights")
    
    model.eval()
    post_processor.eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        if use_post_process:
            output = post_processor(output)
            print("✅ Post-processing was applied")
    
    def denorm_and_bw(t):

        img_rgb = (t.cpu().numpy().transpose(1, 2, 0) + 1) / 2

        gray = 0.299 * img_rgb[...,0] + 0.587 * img_rgb[...,1] + 0.114 * img_rgb[...,2]

        img_bw = np.stack([gray]*3, axis=-1)
        return np.clip(img_bw, 0, 1)
    
    output_bw = denorm_and_bw(output[0])

    if show_result:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(img)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(output_bw)
        axes[1].set_title('Sketch (Black & White)')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()

    if output_path:
        sketch_save = (output_bw * 255).astype(np.uint8)
        Image.fromarray(sketch_save).save(output_path)
        print(f"✅ The sketch has been saved: {output_path}")
    
    return output_bw


def batch_test(image_dir, model_path, output_dir=None, num_samples=5, use_post_process=True):
    
    if output_dir is None:
        output_dir = os.path.join(image_dir, 'sketch_outputs')
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(image_files)} Pictures")
    

    import random
    test_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    for i, img_file in enumerate(test_files):
        print(f"\nProcessing picture {i+1}/{len(test_files)} : {img_file}")
        img_path = os.path.join(image_dir, img_file)
        output_path = os.path.join(output_dir, f'sketch_{img_file}')
        
        test_single_image(img_path, model_path, output_path, show_result=False, use_post_process=use_post_process)
    
    print(f"\n✅ Batch processing complete! Results saved in: {output_dir}")


def visualize_compare(model_path, test_images=None, use_post_process=True):
    
    if test_images is None:

        test_images = [
            "cat.jpg",
            "dog.jpg", 
            "car.jpg",
            "person.jpg"
        ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    post_processor = PostProcessModule().to(device)
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'post_processor_state_dict' in checkpoint and use_post_process:
                post_processor.load_state_dict(checkpoint['post_processor_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    post_processor.eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    fig, axes = plt.subplots(len(test_images), 2, figsize=(10, 4*len(test_images)))
    
    for i, img_name in enumerate(test_images):
        if not os.path.exists(img_name):
            print(f"⚠️ Image does not exist.: {img_name}")
            continue
            
        img = Image.open(img_name).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            if use_post_process:
                output = post_processor(output)
        
        def denorm_and_bw(t):
            img_rgb = (t.cpu().numpy().transpose(1, 2, 0) + 1) / 2
            gray = 0.299 * img_rgb[...,0] + 0.587 * img_rgb[...,1] + 0.114 * img_rgb[...,2]
            return np.stack([gray]*3, axis=-1)
        
        output_bw = denorm_and_bw(output[0])
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Original {i+1}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(output_bw)
        axes[i, 1].set_title(f'Sketch {i+1}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()


def convert_model_format(input_path, output_path):
    
    checkpoint = torch.load(input_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:

        model_weights = checkpoint['model_state_dict']
        torch.save(model_weights, output_path)
        print(f"✅ Converted and saved to: {output_path}")
        
        if 'post_processor_state_dict' in checkpoint:
            post_path = output_path.replace('.pth', '_post.pth')
            torch.save(checkpoint['post_processor_state_dict'], post_path)
            print(f"✅ The post-processor has been saved to: {post_path}")
    else:
        print("❌ The input file is not in dictionary format or does not contain a model_state_dict.")


if __name__ == "__main__":

    MODEL_PATH = r"D:\Final_Year_Project\final-year-project\ai\outputs\photosketch_01375_model.pth"
    TEST_IMAGE = r"D:\Final_Year_Project\final-year-project\ai\test1.jpg"
    OUTPUT_IMAGE = r"D:\Final_Year_Project\final-year-project\ai\outputs\test_sketch.jpg"
    
    print("="*60)
    print("🎨 Test Single picture")
    print("="*60)
    
    test_single_image(TEST_IMAGE, MODEL_PATH, OUTPUT_IMAGE, use_post_process=True)