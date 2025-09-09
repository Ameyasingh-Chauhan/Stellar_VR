import numpy as np
import cv2
from typing import Tuple, List, Optional

def synthesize_stereo(img: np.ndarray, depth_map: np.ndarray, 
                      disparity_px: float = 14.0, 
                      smooth_edges: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create stereo pair from monocular image and depth map for VR180.
    Generates proper left and right eye views with correct disparity.
    
    Args:
        img: Input RGB/BGR image (numpy array)
        depth_map: Depth map (single channel, float32, normalized 0-1)
        disparity_px: Maximum disparity in pixels (controls 3D effect strength)
        smooth_edges: Whether to apply edge smoothing to reduce artifacts
        
    Returns:
        Tuple of (left_eye, right_eye) images ready for VR180
    """
    if img is None or depth_map is None:
        raise ValueError("Input image and depth map are required")
    
    # Ensure inputs are valid
    if img.size == 0 or depth_map.size == 0:
        raise ValueError("Input image or depth map is empty")
    
    # Convert image to float32 for processing
    if img.dtype != np.float32:
        img_float = img.astype(np.float32) / 255.0
    else:
        img_float = img.copy()
    
    # Ensure depth_map is 2D float32
    if depth_map.ndim == 3:
        if depth_map.shape[2] == 3:
            depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
        else:
            depth_map = depth_map[:, :, 0]
    
    if depth_map.dtype != np.float32:
        depth_map = depth_map.astype(np.float32)
    
    # Get dimensions
    h, w = depth_map.shape
    img_h, img_w = img_float.shape[:2]
    
    # Resize depth map to match image if needed
    if h != img_h or w != img_w:
        depth_map = cv2.resize(depth_map, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        h, w = img_h, img_w
    
    # Normalize depth map to [0, 1] range (0=far, 1=near)
    depth_min = np.min(depth_map)
    depth_max = np.max(depth_map)
    
    if depth_max - depth_min < 1e-8:
        # Uniform depth - create slight gradient for some 3D effect
        depth_normalized = np.linspace(0.4, 0.6, h).reshape(-1, 1) * np.ones((h, w), dtype=np.float32)
    else:
        depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
    
    # Apply smoothing to reduce noise while preserving edges
    if smooth_edges:
        # Use bilateral filter to smooth while preserving depth edges
        depth_normalized = cv2.bilateralFilter(
            depth_normalized, d=5, sigmaColor=0.1, sigmaSpace=5
        )
    
    # Calculate disparity map (closer objects have more disparity)
    # For VR180, we want proper stereo separation
    # Closer objects (higher depth values) should have positive disparity
    # Further objects (lower depth values) should have negative disparity
    disparity_map = (depth_normalized - 0.5) * disparity_px
    
    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    
    # Calculate shifted coordinates for left and right eyes
    # Left eye: shift right for objects closer than focal plane (positive disparity)
    # Right eye: shift left for objects closer than focal plane (positive disparity)
    x_left = x_coords + disparity_map * 0.5
    x_right = x_coords - disparity_map * 0.5
    
    # Clamp coordinates to valid image bounds
    x_left = np.clip(x_left, 0, w - 1)
    x_right = np.clip(x_right, 0, w - 1)
    
    # Initialize output images
    left_eye = np.zeros_like(img_float)
    right_eye = np.zeros_like(img_float)
    
    # Apply remapping with high-quality interpolation
    try:
        if len(img_float.shape) == 3:  # Color image
            for c in range(img_float.shape[2]):
                left_eye[..., c] = cv2.remap(
                    img_float[..., c],
                    x_left, y_coords,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101
                )
                right_eye[..., c] = cv2.remap(
                    img_float[..., c],
                    x_right, y_coords,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101
                )
        else:  # Grayscale image
            left_eye = cv2.remap(
                img_float,
                x_left, y_coords,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )
            right_eye = cv2.remap(
                img_float,
                x_right, y_coords,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )
            # Convert grayscale to 3-channel for consistency
            left_eye = cv2.cvtColor(left_eye, cv2.COLOR_GRAY2BGR)
            right_eye = cv2.cvtColor(right_eye, cv2.COLOR_GRAY2BGR)
            
    except Exception as e:
        print(f"Error in stereo remapping: {e}")
        # Fallback: return original image as both eyes
        left_eye = img_float.copy()
        right_eye = img_float.copy()
        
        if len(left_eye.shape) == 2:
            left_eye = cv2.cvtColor(left_eye, cv2.COLOR_GRAY2BGR)
            right_eye = cv2.cvtColor(right_eye, cv2.COLOR_GRAY2BGR)
    
    # Ensure values are in valid range
    left_eye = np.clip(left_eye, 0, 1)
    right_eye = np.clip(right_eye, 0, 1)
    
    # Convert back to uint8
    left_eye_uint8 = (left_eye * 255).astype(np.uint8)
    right_eye_uint8 = (right_eye * 255).astype(np.uint8)
    
    # Ensure 3-channel BGR format
    if len(left_eye_uint8.shape) == 2:
        left_eye_uint8 = cv2.cvtColor(left_eye_uint8, cv2.COLOR_GRAY2BGR)
    if len(right_eye_uint8.shape) == 2:
        right_eye_uint8 = cv2.cvtColor(right_eye_uint8, cv2.COLOR_GRAY2BGR)
    
    # Final validation
    if left_eye_uint8.shape != right_eye_uint8.shape:
        print("Warning: Left and right eye shapes don't match, fixing...")
        min_h = min(left_eye_uint8.shape[0], right_eye_uint8.shape[0])
        min_w = min(left_eye_uint8.shape[1], right_eye_uint8.shape[1])
        left_eye_uint8 = left_eye_uint8[:min_h, :min_w]
        right_eye_uint8 = right_eye_uint8[:min_h, :min_w]
    
    return left_eye_uint8, right_eye_uint8


def enhance_stereo_quality(left: np.ndarray, right: np.ndarray, 
                          contrast_boost: float = 1.1, 
                          saturation_boost: float = 1.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhance stereo pair quality for better VR experience.
    """
    def enhance_image(img):
        if img is None or img.size == 0:
            return img
            
        try:
            # Convert to float32 for processing
            img_float = img.astype(np.float32) / 255.0
            
            # Apply contrast boost
            img_enhanced = np.power(img_float, 1.0 / contrast_boost)
            
            # Apply saturation boost for color images
            if len(img.shape) == 3 and img.shape[2] >= 3:
                # Convert to HSV
                hsv = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2HSV)
                # Boost saturation
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_boost, 0, 1.0)
                # Convert back to BGR
                img_enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Clip and convert back to uint8
            img_enhanced = np.clip(img_enhanced, 0, 1)
            return (img_enhanced * 255).astype(np.uint8)
            
        except Exception as e:
            print(f"Warning: Enhancement failed: {e}")
            return img
    
    return enhance_image(left), enhance_image(right)


def validate_stereo_pair(left: np.ndarray, right: np.ndarray) -> bool:
    """
    Validate that stereo pair is reasonable for VR180.
    """
    if left is None or right is None or left.size == 0 or right.size == 0:
        return False
        
    # Check dimensions match
    if left.shape != right.shape:
        return False
    
    # Check minimum dimensions (must be reasonable for VR)
    h, w = left.shape[:2]
    if h < 360 or w < 640:  # Minimum VR resolution
        return False
    
    # Check if images are too similar (no stereo effect) or too different (broken)
    try:
        # Convert to grayscale for comparison
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if len(left.shape) == 3 else left
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if len(right.shape) == 3 else right
        
        # Calculate correlation
        correlation = cv2.matchTemplate(
            left_gray.astype(np.float32), 
            right_gray.astype(np.float32), 
            cv2.TM_CCOEFF_NORMED
        )[0][0]
        
        # Should have reasonable correlation (not identical, but similar)
        return 0.7 <= abs(correlation) <= 0.98
        
    except Exception:
        return False


def create_vr180_layout(left_eye: np.ndarray, right_eye: np.ndarray) -> np.ndarray:
    """
    Create VR180 side-by-side layout (2:1 aspect ratio).
    """
    if left_eye is None or right_eye is None:
        raise ValueError("Both left and right eye images are required")
    
    # Ensure same dimensions
    if left_eye.shape != right_eye.shape:
        h = min(left_eye.shape[0], right_eye.shape[0])
        w = min(left_eye.shape[1], right_eye.shape[1])
        left_eye = left_eye[:h, :w]
        right_eye = right_eye[:h, :w]
    
    # Create side-by-side layout
    result = np.hstack([left_eye, right_eye])
    
    # Ensure BGR format
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    elif result.shape[2] == 4:
        result = cv2.cvtColor(result, cv2.COLOR_RGBA2BGR)
    
    return result