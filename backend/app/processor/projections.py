# backend/app/processor/projections.py
import cv2
import numpy as np
from typing import Tuple, Optional

def resize_maintain_aspect(img: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    Resize image to target dimensions while maintaining aspect ratio
    and filling the entire target area with proper cropping.
    
    Args:
        img: Input image (numpy array)
        target_width: Target width in pixels
        target_height: Target height in pixels
        
    Returns:
        Resized and cropped image
    """
    if img is None or target_width <= 0 or target_height <= 0:
        raise ValueError("Invalid input parameters")
        
    h, w = img.shape[:2]
    
    # Calculate scale factors
    scale_w = target_width / w
    scale_h = target_height / h
    
    # Use the larger scale to ensure we fill the entire target area
    scale = max(scale_w, scale_h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image with high-quality interpolation
    resized = cv2.resize(img, (new_w, new_h), 
                        interpolation=cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA)
    
    # Calculate crop offsets to center the image
    start_x = max(0, (new_w - target_width) // 2)
    start_y = max(0, (new_h - target_height) // 2)
    
    # Ensure we don't go out of bounds
    start_x = min(start_x, new_w - target_width)
    start_y = min(start_y, new_h - target_height)
    
    # Crop to exact target size
    cropped = resized[start_y:start_y + target_height, start_x:start_x + target_width]
    
    # Final size check
    if cropped.shape[0] != target_height or cropped.shape[1] != target_width:
        cropped = cv2.resize(cropped, (target_width, target_height), 
                            interpolation=cv2.INTER_LINEAR)
    
    return cropped

def create_vr180_layout(left_eye: np.ndarray, right_eye: np.ndarray, 
                       layout: str = "left_right") -> np.ndarray:
    """
    Create proper VR 180 stereo layout in rectangular format.
    
    Args:
        left_eye: Left eye view (numpy array)
        right_eye: Right eye view (numpy array)
        layout: Layout type - "left_right" (2:1) or "top_bottom" (1:2)
        
    Returns:
        Combined stereo image in the specified layout
        
    Raises:
        ValueError: If input images are invalid or have mismatched dimensions
    """
    if left_eye is None or right_eye is None:
        raise ValueError("Both left and right eye images are required")
        
    # Ensure both eyes have the same dimensions
    if left_eye.shape != right_eye.shape:
        # Find common dimensions
        h = min(left_eye.shape[0], right_eye.shape[0])
        w = min(left_eye.shape[1], right_eye.shape[1])
        
        # Crop both images to common dimensions
        left_eye = left_eye[:h, :w]
        right_eye = right_eye[:h, :w]
        
        # If still not matching, resize to the smallest dimensions
        if left_eye.shape != right_eye.shape:
            min_h = min(left_eye.shape[0], right_eye.shape[0])
            min_w = min(left_eye.shape[1], right_eye.shape[1])
            left_eye = cv2.resize(left_eye, (min_w, min_h), interpolation=cv2.INTER_LINEAR)
            right_eye = cv2.resize(right_eye, (min_w, min_h), interpolation=cv2.INTER_LINEAR)
    
    # Create the appropriate layout
    if layout == "left_right":
        # Side-by-side (2:1 aspect ratio) - standard for VR180
        result = np.hstack([left_eye, right_eye])
    elif layout == "top_bottom":
        # Top-bottom (1:2 aspect ratio) - alternative layout
        result = np.vstack([left_eye, right_eye])
    else:
        raise ValueError(f"Unsupported layout: {layout}. Use 'left_right' or 'top_bottom'")
    
    # Ensure the result has the correct number of color channels
    if len(result.shape) == 2:  # Grayscale
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    elif result.shape[2] == 4:  # RGBA
        result = cv2.cvtColor(result, cv2.COLOR_RGBA2BGR)
    elif result.shape[2] == 1:  # Single channel
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
    return result

def prepare_eye_images(left_img: np.ndarray, right_img: np.ndarray, 
                      target_eye_width: int, target_eye_height: int, 
                      layout: str = "left_right") -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare individual eye images with proper dimensions for VR 180.
    
    Args:
        left_img: Left eye image
        right_img: Right eye image
        target_eye_width: Target width for each eye view
        target_eye_height: Target height for each eye view
        layout: Layout type - "left_right" or "top_bottom"
        
    Returns:
        Tuple of (left_eye, right_eye) processed images
        
    Raises:
        ValueError: If input images are invalid or processing fails
    """
    if left_img is None or right_img is None:
        raise ValueError("Both left and right eye images are required")
        
    if target_eye_width <= 0 or target_eye_height <= 0:
        raise ValueError("Target dimensions must be positive")
    
    try:
        # Resize both eyes to target dimensions while maintaining aspect ratio
        left_eye = resize_maintain_aspect(left_img, target_eye_width, target_eye_height)
        right_eye = resize_maintain_aspect(right_img, target_eye_width, target_eye_height)
        
        # Ensure both eyes have exactly the same dimensions
        h = min(left_eye.shape[0], right_eye.shape[0])
        w = min(left_eye.shape[1], right_eye.shape[1])
        
        # If images are different sizes, crop to the smallest common size
        if h != left_eye.shape[0] or w != left_eye.shape[1]:
            left_eye = left_eye[:h, :w]
        if h != right_eye.shape[0] or w != right_eye.shape[1]:
            right_eye = right_eye[:h, :w]
            
        # Final resize to ensure exact dimensions
        if left_eye.shape[0] != target_eye_height or left_eye.shape[1] != target_eye_width:
            left_eye = cv2.resize(left_eye, (target_eye_width, target_eye_height), 
                                interpolation=cv2.INTER_LINEAR)
        if right_eye.shape[0] != target_eye_height or right_eye.shape[1] != target_eye_width:
            right_eye = cv2.resize(right_eye, (target_eye_width, target_eye_height), 
                                 interpolation=cv2.INTER_LINEAR)
        
        return left_eye, right_eye
        
    except Exception as e:
        raise RuntimeError(f"Failed to prepare eye images: {str(e)}")

def validate_vr180_dimensions(img: np.ndarray, layout: str = "left_right") -> bool:
    """
    Validate that the final VR 180 video has correct aspect ratio and dimensions.
    
    Args:
        img: Image to validate
        layout: Layout type - "left_right" or "top_bottom"
        
    Returns:
        bool: True if dimensions are valid, False otherwise
    """
    if img is None:
        return False
        
    h, w = img.shape[:2]
    
    # Check minimum dimensions
    min_dimension = 480  # Minimum dimension for VR180
    if w < min_dimension or h < min_dimension:
        return False
    
    # VR 180 standard aspect ratios:
    # - Side-by-side: 2:1 (width:height) for left_right
    # - Top-bottom: 1:2 (width:height) for top_bottom
    if layout == "left_right":
        target_ratio = 2.0  # 2:1 aspect ratio
        min_acceptable_ratio = 1.8  # 10% tolerance
        max_acceptable_ratio = 2.2  # 10% tolerance
    else:  # top_bottom
        target_ratio = 0.5  # 1:2 aspect ratio
        min_acceptable_ratio = 0.45  # 10% tolerance
        max_acceptable_ratio = 0.55  # 10% tolerance
    
    actual_ratio = w / h
    
    # Check if the aspect ratio is within acceptable bounds
    if not (min_acceptable_ratio <= actual_ratio <= max_acceptable_ratio):
        return False
    
    # Check if dimensions are even numbers (required by some codecs)
    if w % 2 != 0 or h % 2 != 0:
        return False
    
    # Check if resolution is standard (e.g., 1920x960, 3840x1920, etc.)
    standard_heights = [540, 720, 1080, 1440, 2160, 2880, 4320]
    standard_widths = [h * 2 for h in standard_heights]  # For 2:1 aspect ratio
    
    if layout == "left_right":
        if h not in standard_heights or w != h * 2:
            # Not a standard resolution, but might still be valid
            pass
    else:  # top_bottom
        if h != w * 2:
            return False
    
    return True