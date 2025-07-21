import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2

# Add parent directory to path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from haven import haven_utils as hu
from src import models
from src.datasets import jcu_fish

class SegmentationModel:
    def __init__(self, exp_name='a106acdd252ec8c131d81b70a2014ffc'):
        self.exp_name = exp_name
        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        self.exp_dict_path = os.path.join(self.base_dir, 'results', exp_name, 'exp_dict.json')
        self.model_path = os.path.join(self.base_dir, 'results', exp_name, 'model_best.pth')
        
        self.model = None
        self.transform = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and transformations"""
        # Load experiment configuration
        exp_dict = hu.load_json(self.exp_dict_path)
        
        # Create dummy dataset to get transformations
        dummy_dataset = jcu_fish.JcuFish(
            split="val",
            datadir='DeepAgro',
            exp_dict=exp_dict
        )
        
        self.transform = dummy_dataset.img_transform
        
        # Load model
        self.model = models.get_model(
            model_dict=exp_dict['model'], 
            exp_dict=exp_dict, 
            train_set=dummy_dataset
        )
        self.model.load_state_dict(hu.torch_load(self.model_path))
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        self.model.eval()
    
    def predict_image(self, image, inference_type="full"):
        """
        Predict on image
        Args:
            image: PIL Image
            inference_type: "full" or "bottom_half"
        Returns:
            prediction: numpy array with probabilities [0-1]
        """
        original_size = image.size
        
        # Process image based on inference type
        if inference_type == "bottom_half":
            # Crop bottom half
            width, height = image.size
            image = image.crop((0, height//2, width, height))
        
        # Apply transformations
        image_tensor = self.transform(image).unsqueeze(0)
        
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        
        # Predict
        with torch.no_grad():
            batch = {
                'images': image_tensor,
                'meta': [{'shape': image_tensor.shape[-2:]}]
            }
            pred = self.model.predict_on_batch(batch)
        
        prediction = pred[0]  # Get first (and only) prediction
        
        # Debug: Print prediction info
        print(f"Prediction type: {type(prediction)}")
        print(f"Prediction shape (if available): {getattr(prediction, 'shape', 'No shape attribute')}")
        
        # Convert to numpy if it's a tensor
        if torch.is_tensor(prediction):
            if prediction.is_cuda:
                prediction = prediction.cpu()
            prediction = prediction.numpy()
        
        # Check if prediction is valid
        if prediction is None:
            raise ValueError("Model returned None prediction")
        
        # Ensure prediction is numpy array
        if not isinstance(prediction, np.ndarray):
            prediction = np.array(prediction)
        
        print(f"After conversion - shape: {prediction.shape}, dtype: {prediction.dtype}")
        
        # Ensure prediction is float32 and 2D
        prediction = prediction.astype(np.float32)
        
        # Handle different possible shapes
        if len(prediction.shape) > 2:
            # If 3D or 4D, squeeze and take first channel if needed
            prediction = np.squeeze(prediction)
            if len(prediction.shape) > 2:
                prediction = prediction[0] if prediction.shape[0] == 1 else prediction[:, :, 0]
        elif len(prediction.shape) == 1:
            # If 1D, reshape to square if possible
            size = int(np.sqrt(len(prediction)))
            if size * size == len(prediction):
                prediction = prediction.reshape(size, size)
            else:
                raise ValueError(f"Cannot reshape 1D prediction of length {len(prediction)} to 2D")
        
        print(f"Final prediction shape: {prediction.shape}")
        
        # Ensure we have a valid 2D array
        if len(prediction.shape) != 2:
            raise ValueError(f"Expected 2D prediction, got shape: {prediction.shape}")
        
        # Resize prediction back to original size if needed
        if inference_type == "bottom_half":
            # Create full-size prediction array
            full_pred = np.zeros((original_size[1], original_size[0]), dtype=np.float32)
            
            # Calculate exact dimensions for bottom half
            start_row = original_size[1] // 2
            bottom_half_height = original_size[1] - start_row  # This handles odd heights correctly
            
            # Resize the cropped prediction to match the actual bottom half dimensions
            pred_resized = cv2.resize(prediction, (original_size[0], bottom_half_height), interpolation=cv2.INTER_LINEAR)
            
            # Assign to the correct slice
            full_pred[start_row:, :] = pred_resized
            prediction = full_pred
        else:
            # Resize to original dimensions
            prediction = cv2.resize(prediction, original_size, interpolation=cv2.INTER_LINEAR)
        
        # Ensure values are in [0, 1] range
        prediction = np.clip(prediction, 0, 1)
        
        return prediction
    
    def count_objects(self, mask):
        """Count objects using connected components with proximity-based merging"""
        # Convert to binary mask
        binary_mask = (mask > 0).astype(np.uint8) * 255
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        
        if num_labels <= 1:  # Only background
            return 0, labels, np.array([])
        
        # Extract component information (excluding background at index 0)
        areas = stats[1:, cv2.CC_STAT_AREA]  # Area of each component
        centroids_clean = centroids[1:]  # Centroids excluding background
        
        # Create list of components with their properties
        components = []
        for i in range(len(areas)):
            components.append({
                'index': i + 1,  # Original label index (1-based)
                'area': areas[i],
                'centroid': centroids_clean[i],
                'merged': False  # Track if component has been merged
            })
        
        # Sort components by area (largest first)
        components.sort(key=lambda x: x['area'], reverse=True)
        
        # Merge small components with nearby large ones
        size_ratio_threshold = 10  # Small component should be 20x smaller
        proximity_threshold = 15  # Maximum distance in pixels
        
        for i, large_comp in enumerate(components):
            if large_comp['merged']:
                continue
                
            for j, small_comp in enumerate(components[i+1:], i+1):
                if small_comp['merged']:
                    continue
                
                # Check size ratio
                size_ratio = large_comp['area'] / small_comp['area']
                if size_ratio < size_ratio_threshold:
                    continue
                
                # Check proximity
                dist = np.sqrt(
                    (large_comp['centroid'][0] - small_comp['centroid'][0])**2 +
                    (large_comp['centroid'][1] - small_comp['centroid'][1])**2
                )
                
                if dist <= proximity_threshold:
                    # Merge: mark small component as merged
                    small_comp['merged'] = True
                    print(f"Merged component {small_comp['index']} (area: {small_comp['area']}) "
                          f"into component {large_comp['index']} (area: {large_comp['area']}) "
                          f"- distance: {dist:.1f}, ratio: {size_ratio:.1f}")
        
        # Count non-merged components
        final_components = [comp for comp in components if not comp['merged']]
        object_count = len(final_components)
        
        # Create final centroids array
        final_centroids = np.array([comp['centroid'] for comp in final_components])
        
        return object_count, labels, final_centroids
    
    def calculate_position_metrics(self, centroids, mask_shape):
        """
        Calculate positioning metrics for quality control
        
        This function defines a "valid region" (excluding borders) and counts how many
        detected objects fall outside this region. This helps identify:
        
        1. Objects too close to image edges (potentially cut off)
        2. Noise or artifacts near borders
        3. Overall quality of detection positioning
        
        Args:
            centroids: Array of object centroids [(x, y), ...]
            mask_shape: Shape of the mask (height, width)
            
        Returns:
            dict with metrics:
            - objects_outside_region: Count of objects in border areas
            - total_objects: Total detected objects
            - percentage_outside: Percentage of objects in border areas
        """
        if len(centroids) == 0:
            return {
                "objects_outside_region": 0, 
                "total_objects": 0, 
                "percentage_outside": 0.0,
                "border_size_pixels": 0
            }
        
        height, width = mask_shape
        
        # Define border exclusion zone (10% of the smaller dimension)
        border_size = min(width, height) * 0.1
        valid_region = {
            'x_min': border_size,
            'x_max': width - border_size,
            'y_min': border_size, 
            'y_max': height - border_size
        }
        
        objects_outside = 0
        for centroid in centroids:
            x, y = centroid
            if not (valid_region['x_min'] <= x <= valid_region['x_max'] and 
                    valid_region['y_min'] <= y <= valid_region['y_max']):
                objects_outside += 1
        
        return {
            "objects_outside_region": objects_outside,
            "total_objects": len(centroids),
            "percentage_outside": (objects_outside / len(centroids)) * 100 if len(centroids) > 0 else 0,
            "border_size_pixels": int(border_size)
        }
        
