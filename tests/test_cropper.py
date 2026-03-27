# tests/test_cropper.py
import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from PIL import Image
from app import SaliencyCropper


class TestSaliencyCropper(unittest.TestCase):
    """Image cropper unit tests"""
    
    def setUp(self):
        self.cropper = SaliencyCropper()
        self.test_img = Image.new('RGB', (400, 400), color='red')
        self.wide_img = Image.new('RGB', (800, 400), color='blue')
        self.tall_img = Image.new('RGB', (400, 800), color='green')
    
    def test_center_crop_1x1_square(self):
        """Test: 1:1 crop on square image"""
        result, info = self.cropper.process(self.test_img, "1:1")
        
        self.assertEqual(result.size[0], result.size[1])
        self.assertEqual(info['method'], 'center_crop')
    
    def test_center_crop_16x9_wide(self):
        """Test: 16:9 crop on wide image"""
        result, info = self.cropper.process(self.wide_img, "16:9")
        
        w, h = result.size
        expected_ratio = 16 / 9
        actual_ratio = w / h
        self.assertAlmostEqual(actual_ratio, expected_ratio, delta=0.01)
    
    def test_center_crop_9x16_tall(self):
        """Test: 9:16 crop on tall image"""
        result, info = self.cropper.process(self.tall_img, "9:16")
        
        w, h = result.size
        expected_ratio = 9 / 16
        actual_ratio = w / h
        self.assertAlmostEqual(actual_ratio, expected_ratio, delta=0.01)
    
    def test_crop_returns_info_dict(self):
        """Test: Returned info dict contains required fields"""
        result, info = self.cropper.process(self.test_img, "1:1")
        
        self.assertIn('method', info)
        self.assertIn('processing_time', info)
        self.assertIn('original_size', info)
        self.assertIn('crop_box', info)
    
    def test_crop_box_is_ints(self):
        """Test: Crop box coordinates are integers"""
        result, info = self.cropper.process(self.test_img, "1:1")
        
        crop_box = info['crop_box']
        for val in crop_box:
            self.assertIsInstance(val, int)
    
    def test_stats_increment(self):
        """Test: Statistics counter increments correctly"""
        initial_stats = self.cropper.get_stats()['total']
        self.cropper.process(self.test_img, "1:1")
        new_stats = self.cropper.get_stats()['total']
        
        self.assertEqual(new_stats, initial_stats + 1)


if __name__ == '__main__':
    unittest.main()