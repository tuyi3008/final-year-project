# test_model.py - Test the trained model

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# Import model
from model import StyleTransferModel

def test_trained_model():
    print("🧪 Testing trained model...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load trained model
    model_path = '../outputs/final_model.pth'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Create model
    model = StyleTransferModel(num_channels=32, num_residual_blocks=5)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ Model loaded from: {model_path}")
    
    # Load a test image
    content_dir = '../../datasets/content'
    content_images = [f for f in os.listdir(content_dir) if f.endswith(('.png', '.jpg'))]
    
    if not content_images:
        print("❌ No test images found")
        return
    
    # Use first content image
    test_image_path = os.path.join(content_dir, content_images[0])
    print(f"Test image: {test_image_path}")
    
    # Load and transform image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load image
    image = Image.open(test_image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Run model
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    
    # Convert tensors back to images
    def tensor_to_image(tensor):
        img = tensor.cpu().squeeze(0)  # Remove batch dimension
        img = (img + 1) / 2  # [-1, 1] to [0, 1]
        img = transforms.ToPILImage()(img)
        return img
    
    # Create comparison
    original_img = image
    stylized_img = tensor_to_image(output_tensor)
    
    # Display images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(stylized_img)
    axes[1].set_title('Stylized Image')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save result
    result_path = '../outputs/test_result.png'
    plt.savefig(result_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Test result saved to: {result_path}")
    print("Note: Since we only trained for 2 epochs with random images,")
    print("the style transfer effect might be subtle.")

if __name__ == "__main__":
    test_trained_model()
