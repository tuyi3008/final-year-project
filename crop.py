# crop.py
import time
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict, List

# =============================
# Saliency-based Cropper Class - ULTRA FAST VERSION
# =============================
class SaliencyCropper:
    """
    Fast cropping based on face detection and center crop
    Removed slow saliency detection for speed
    """
    
    def __init__(self):
        # Initialize face detector only (remove saliency)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Statistics
        self.stats = {
            'total': 0,
            'face_detected': 0,
            'center_used': 0,
            'fallback_used': 0
        }
    
    def process(self, 
                pil_image: Image.Image, 
                target_ratio: str = "1:1") -> Tuple[Image.Image, Dict]:
        """
        Ultra fast processing - no saliency detection
        """
        start_time = time.time()
        self.stats['total'] += 1
        
        # Record processing info
        info = {
            'method': 'unknown',
            'face_count': 0,
            'processing_time': 0,
            'crop_box': None,
            'original_size': pil_image.size
        }
        
        try:
            # Convert format for face detection
            img_np = np.array(pil_image)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            h, w = img_cv.shape[:2]
            
            # Parse target ratio
            target_w, target_h = map(int, target_ratio.split(':'))
            target_ratio_float = target_w / target_h
            
            # Fast face detection (less sensitive for speed)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.3,        # Increased for fewer false positives
                minNeighbors=8,          # Increased for fewer false positives
                minSize=(60, 60)         # Increased to ignore small false detections
            )
            
            info['face_count'] = len(faces)
            
            # Decision: use face detection or center crop
            if len(faces) > 0:
                # Use face-protected crop
                crop_box = self._face_protected_crop(
                    faces, (w, h), target_ratio_float
                )
                info['method'] = 'face_protected'
                self.stats['face_detected'] += 1
            else:
                # Use center crop
                crop_box = self._get_center_crop_box(
                    (w, h), target_ratio_float
                )
                info['method'] = 'center_crop'
                self.stats['center_used'] += 1
            
            # FIX: Convert numpy ints to Python ints for JSON serialization
            if crop_box:
                info['crop_box'] = tuple(int(x) for x in crop_box)
            
            # Execute crop
            x1, y1, x2, y2 = crop_box
            cropped = pil_image.crop((x1, y1, x2, y2))
            
            info['processing_time'] = time.time() - start_time
            
            return cropped, info
            
        except Exception as e:
            print(f"Processing error: {e}, using fallback")
            info['method'] = 'error_fallback'
            self.stats['fallback_used'] += 1
            # Simple center crop fallback
            result = self._center_crop(pil_image, target_ratio_float)
            info['processing_time'] = time.time() - start_time
            return result, info
    
    def _face_protected_crop(self, 
                            faces: List, 
                            img_size: Tuple[int, int], 
                            target_ratio: float) -> Tuple[int, int, int, int]:
        """
        Face-protected cropping - returns Python ints
        """
        w, h = img_size
        
        # Calculate bounding box containing all faces
        min_x = min(x for (x, y, fw, fh) in faces)
        max_x = max(x + fw for (x, y, fw, fh) in faces)
        min_y = min(y for (x, y, fw, fh) in faces)
        max_y = max(y + fh for (x, y, fw, fh) in faces)
        
        # Expand 20% as buffer
        padding = 0.2
        width_pad = int((max_x - min_x) * padding)
        height_pad = int((max_y - min_y) * padding)
        
        face_center_x = (min_x + max_x) // 2
        face_center_y = (min_y + max_y) // 2
        
        # Adjust according to target ratio
        if w / h > target_ratio:
            crop_w = int(h * target_ratio)
            left = max(0, min(face_center_x - crop_w // 2, w - crop_w))
            # FIX: Ensure all values are Python ints
            return (int(left), 0, int(left + crop_w), int(h))
        else:
            crop_h = int(w / target_ratio)
            top = max(0, min(face_center_y - crop_h // 2, h - crop_h))
            # FIX: Ensure all values are Python ints
            return (0, int(top), int(w), int(top + crop_h))
    
    def _get_center_crop_box(self, 
                            img_size: Tuple[int, int], 
                            target_ratio: float) -> Tuple[int, int, int, int]:
        """Get center crop box - returns Python ints"""
        w, h = img_size
        
        if w / h > target_ratio:
            crop_w = int(h * target_ratio)
            left = (w - crop_w) // 2
            # FIX: Ensure all values are Python ints
            return (int(left), 0, int(left + crop_w), int(h))
        else:
            crop_h = int(w / target_ratio)
            top = (h - crop_h) // 2
            # FIX: Ensure all values are Python ints
            return (0, int(top), int(w), int(top + crop_h))
    
    def _center_crop(self, 
                    pil_image: Image.Image, 
                    target_ratio: float) -> Image.Image:
        """Center crop - returns cropped image"""
        w, h = pil_image.size
        
        if w / h > target_ratio:
            new_w = int(h * target_ratio)
            left = (w - new_w) // 2
            return pil_image.crop((left, 0, left + new_w, h))
        else:
            new_h = int(w / target_ratio)
            top = (h - new_h) // 2
            return pil_image.crop((0, top, w, top + new_h))
    
    def get_stats(self) -> Dict:
        """Get statistics"""
        return self.stats


# =============================
# Initialize SaliencyCropper
# =============================
cropper = SaliencyCropper()