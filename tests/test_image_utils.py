# tests/test_image_utils.py
import unittest
import torch
from PIL import Image
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import image_to_tensor, tensor_to_pil


class TestImageUtils(unittest.TestCase):
    """Image processing utility unit tests"""
    
    def setUp(self):
        self.test_img = Image.new('RGB', (256, 256), color='red')
    
    def test_image_to_tensor_returns_tensor(self):
        """Test: Image to tensor returns torch.Tensor"""
        tensor = image_to_tensor(self.test_img)
        self.assertIsInstance(tensor, torch.Tensor)
    
    def test_image_to_tensor_shape(self):
        """Test: Tensor shape is (batch, channels, height, width)"""
        tensor = image_to_tensor(self.test_img)
        
        self.assertEqual(tensor.dim(), 4)
        self.assertEqual(tensor.shape[0], 1)
        self.assertEqual(tensor.shape[1], 3)
        self.assertEqual(tensor.shape[2], 256)
        self.assertEqual(tensor.shape[3], 256)
    
    def test_image_to_tensor_normalization(self):
        """Test: Tensor values are in [-1, 1] range after normalization"""
        tensor = image_to_tensor(self.test_img)
        
        self.assertGreaterEqual(tensor.min(), -1.0)
        self.assertLessEqual(tensor.max(), 1.0)
    
    def test_tensor_to_pil_returns_image(self):
        """Test: Tensor to PIL returns Image object"""
        mock_tensor = torch.randn(1, 3, 256, 256)
        result = tensor_to_pil(mock_tensor, style="anime")
        self.assertIsInstance(result, Image.Image)
    
    def test_tensor_to_pil_sketch_grayscale(self):
        """Test: Sketch style output is grayscale"""
        mock_tensor = torch.randn(1, 3, 256, 256)
        result = tensor_to_pil(mock_tensor, style="sketch")
        
        img_array = np.array(result.convert('RGB'))
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        
        self.assertTrue(np.allclose(r, g, atol=2))
        self.assertTrue(np.allclose(g, b, atol=2))
    
    def test_tensor_to_pil_cyberpunk_style(self):
        """Test: Cyberpunk style handles tanh output"""
        mock_tensor = torch.tanh(torch.randn(1, 3, 64, 64))
        result = tensor_to_pil(mock_tensor, style="cyberpunk")
        self.assertIsInstance(result, Image.Image)
    
    def test_tensor_to_pil_ukiyoe_style(self):
        """Test: Ukiyo-e style handles tanh output"""
        mock_tensor = torch.tanh(torch.randn(1, 3, 64, 64))
        result = tensor_to_pil(mock_tensor, style="ukiyoe")
        self.assertIsInstance(result, Image.Image)
    
    def test_tensor_to_pil_anime_style(self):
        """Test: Anime style works with random tensor"""
        mock_tensor = torch.randn(1, 3, 64, 64)
        result = tensor_to_pil(mock_tensor, style="anime")
        self.assertIsInstance(result, Image.Image)
    
    def test_tensor_to_pil_handles_extreme_values(self):
        """Test: Extreme values do not cause crashes"""
        mock_tensor = torch.tensor([[[[100.0]]], [[[-100.0]]], [[[200.0]]]])
        mock_tensor = mock_tensor.reshape(1, 3, 1, 1)
        
        try:
            result = tensor_to_pil(mock_tensor, style="anime")
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"tensor_to_pil raised an exception: {e}")
    
    def test_tensor_to_pil_output_range(self):
        """Test: Output pixel values are within valid range"""
        mock_tensor = torch.randn(1, 3, 64, 64)
        result = tensor_to_pil(mock_tensor, style="anime")
        
        img_array = np.array(result)
        self.assertGreaterEqual(img_array.min(), 0)
        self.assertLessEqual(img_array.max(), 255)


if __name__ == '__main__':
    unittest.main()