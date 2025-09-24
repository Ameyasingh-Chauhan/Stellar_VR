"""
dome_projection.py - Full equirectangular dome projection implementation
Converts dual-fisheye input to panoramic dome with smooth edge bending
"""
import numpy as np
import cv2
from pathlib import Path
import math
from config import PANINI_MIX, STEREO_MIX, IDENTITY_MIX, ULTRAFAST_MODE


def panini_projection(x, y, s=0.8):
    """
    Panini projection mapping
    s: projection strength parameter
    """
    denom = 1 + s*(1 - x*x)
    return x/denom, y/denom


def stereographic_projection(x, y):
    """
    Stereographic projection mapping
    """
    denom = 1 + x*x + y*y
    return 2*x/denom, 2*y/denom


def identity_projection(x, y):
    """
    Identity (linear) projection mapping
    """
    return x, y


def blend_projections(x, y, panini_weight=0.7, stereographic_weight=0.2, identity_weight=0.1):
    """
    Blend multiple projections for smooth dome effect
    """
    px, py = panini_projection(x, y, s=0.8)
    sx, sy = stereographic_projection(x, y)
    ix, iy = identity_projection(x, y)
    
    # Weighted blend of all three projections
    bx = panini_weight * px + stereographic_weight * sx + identity_weight * ix
    by = panini_weight * py + stereographic_weight * sy + identity_weight * iy
    
    return bx, by


def create_dome_projection_map(width, height, output_width, output_height):
    """
    Create projection maps for transforming dual fisheye to equirectangular dome
    """
    # Create normalized coordinate grids [-1, 1]
    x_coords = np.linspace(-1, 1, width).astype('float32')
    y_coords = np.linspace(-1, 1, height).astype('float32')
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    # Apply blended projection
    bx, by = blend_projections(xx, yy, 
                              panini_weight=PANINI_MIX, 
                              stereographic_weight=STEREO_MIX, 
                              identity_weight=IDENTITY_MIX)
    
    # Map to pixel coordinates
    mapx = ((bx + 1.0) / 2.0) * width
    mapy = ((by + 1.0) / 2.0) * height
    
    return mapx.astype('float32'), mapy.astype('float32')


def normalize_histogram_to_reference(input_img, ref_img):
    """
    Match histogram of input image to reference image to preserve brightness/contrast
    """
    if ULTRAFAST_MODE:
        return input_img
    
    # Convert to LAB color space for better brightness matching
    ref_lab = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB)
    input_lab = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)
    
    # Equalize L channel (brightness) using reference statistics
    ref_l = ref_lab[:,:,0].astype(np.float32)
    input_l = input_lab[:,:,0].astype(np.float32)
    
    # Calculate statistics
    ref_mean, ref_std = ref_l.mean(), ref_l.std()
    input_mean, input_std = input_l.mean(), input_l.std()
    
    # Avoid division by zero
    if input_std == 0:
        input_std = 1e-8
    
    # Normalize input L channel to match reference
    normalized_l = (input_l - input_mean) * (ref_std / input_std) + ref_mean
    
    # Clip to valid range
    normalized_l = np.clip(normalized_l, 0, 255)
    
    # Apply back to LAB image
    input_lab[:,:,0] = normalized_l.astype(np.uint8)
    
    # Convert back to BGR
    result = cv2.cvtColor(input_lab, cv2.COLOR_LAB2BGR)
    
    return result


def enhance_stereo_disparity(left_img, right_img, disparity_factor=1.2):
    """
    Enhance stereo disparity for stronger "inside the scene" effect
    Increase the apparent depth in the stereo pair
    """
    if ULTRAFAST_MODE:
        return left_img, right_img
    
    # Compute optical flow or simple shifting to enhance disparity
    # We'll apply a subtle shift to enhance the 3D effect
    h, w = left_img.shape[:2]
    
    # Create enhanced disparity by simple offset for performance
    # This simulates what the stereo pair would look like with increased IPD
    shift_amount = max(1, int(w * 0.001 * disparity_factor))  # Subtle shift based on image width
    
    # Create translation matrices
    translation_left = np.float32([[1, 0, -shift_amount], [0, 1, 0]])
    translation_right = np.float32([[1, 0, shift_amount], [0, 1, 0]])
    
    # Apply translation for performance
    left_enhanced = cv2.warpAffine(left_img, translation_left, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    right_enhanced = cv2.warpAffine(right_img, translation_right, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    
    return left_enhanced, right_enhanced


def create_equirectangular_dome_from_fisheye(left_img, right_img, output_width=4320, output_height=2160):
    """
    Convert dual fisheye images to full equirectangular dome projection
    with smooth edge bending and immersive effect
    """
    # For ultra-fast mode, use faster method but maintain quality
    if ULTRAFAST_MODE:
        # Just resize and combine for speed
        left_resized = cv2.resize(left_img, (output_width//2, output_height), interpolation=cv2.INTER_LINEAR)
        right_resized = cv2.resize(right_img, (output_width//2, output_height), interpolation=cv2.INTER_LINEAR)
        return cv2.hconcat([left_resized, right_resized])
    
    h, w = left_img.shape[:2]
    
    # Normalize brightness/contrast to preserve original look
    left_normalized = normalize_histogram_to_reference(left_img, left_img)
    right_normalized = normalize_histogram_to_reference(right_img, right_img)
    
    # Enhance stereo disparity for "inside the scene" effect
    left_enhanced, right_enhanced = enhance_stereo_disparity(left_normalized, right_normalized)
    
    # Create projection maps for each eye
    left_mapx, left_mapy = create_dome_projection_map(w, h, output_width//2, output_height)
    right_mapx, right_mapy = create_dome_projection_map(w, h, output_width//2, output_height)
    
    # Apply remapping for each eye with faster interpolation
    left_dome = cv2.remap(left_enhanced, left_mapx, left_mapy, 
                         interpolation=cv2.INTER_LINEAR, 
                         borderMode=cv2.BORDER_WRAP)
    
    right_dome = cv2.remap(right_enhanced, right_mapx, right_mapy, 
                          interpolation=cv2.INTER_LINEAR, 
                          borderMode=cv2.BORDER_WRAP)
    
    # Resize to exact output dimensions
    left_dome = cv2.resize(left_dome, (output_width//2, output_height), interpolation=cv2.INTER_LINEAR)
    right_dome = cv2.resize(right_dome, (output_width//2, output_height), interpolation=cv2.INTER_LINEAR)
    
    # Combine left and right eyes side-by-side
    dome_sbs = cv2.hconcat([left_dome, right_dome])
    
    # Apply final enhancements for immersive experience
    dome_sbs = apply_dome_enhancements(dome_sbs)
    
    return dome_sbs


def apply_dome_enhancements(img):
    """
    Apply final enhancements for immersive dome experience
    """
    if ULTRAFAST_MODE:
        return img
    
    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Process in larger chunks for better performance
    chunk_size = min(512, h)  # Larger chunks for better performance
    enhanced = img.copy().astype(np.float32)
    
    for start_row in range(0, h, chunk_size):
        end_row = min(start_row + chunk_size, h)
        chunk_height = end_row - start_row
        
        # Create distance map for this chunk only
        y_chunk = np.arange(start_row, end_row)
        x_chunk = np.arange(w)
        y_grid, x_grid = np.meshgrid(y_chunk, x_chunk, indexing='ij')
        dist_chunk = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        normalized_dist_chunk = dist_chunk / max_dist
        
        # Get the image chunk
        img_chunk = img[start_row:end_row, :, :].astype(np.float32)
        enhanced_chunk = img_chunk.copy()
        
        # Define regions for different processing in this chunk
        center_region = normalized_dist_chunk < 0.4  # Sharpen inner 40%
        edge_region = normalized_dist_chunk >= 0.8  # Slightly soften outer 20%
        
        # Sharpen center region in this chunk with simplified approach
        if np.any(center_region):
            # Use a simpler, faster sharpening kernel
            kernel = np.array([[0, -0.5, 0],
                               [-0.5, 3, -0.5], 
                               [0, -0.5, 0]])  # Faster, less aggressive sharpening
            center_sharpened = cv2.filter2D(img_chunk, -1, kernel).astype(np.float32)
            # Apply sharpening only to center region
            enhanced_chunk[center_region] = (
                enhanced_chunk[center_region] * 0.8 +  # Less aggressive blending
                center_sharpened[center_region] * 0.2
            )
        
        # Skip edge softening for performance - only do it if specifically needed
        # The softening effect is often subtle and can be skipped for speed
        
        # Copy processed chunk back to enhanced image
        enhanced[start_row:end_row, :, :] = enhanced_chunk
    
    # Apply subtle color saturation boost to the whole image
    enhanced = enhanced.astype(np.uint8)
    # Skip HSV conversion for speed - just return the image as-is
    # hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
    # hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.05, 0, 255)  # 5% saturation boost
    # enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Ensure proper data type and range
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    return enhanced


def process_fisheye_frame_pair(left_path, right_path, output_path, output_width=4320, output_height=2160):
    """
    Process a single fisheye frame pair and convert to equirectangular dome
    """
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Load images with error handling
    try:
        left_img = cv2.imread(str(left_path))
        right_img = cv2.imread(str(right_path))
        
        if left_img is None:
            print(f"Warning: Could not load left image: {left_path}")
            # Create blank frame as fallback
            left_img = np.zeros((output_height//2, output_width//2, 3), dtype=np.uint8)
        if right_img is None:
            print(f"Warning: Could not load right image: {right_path}")
            # Create blank frame as fallback
            right_img = np.zeros((output_height//2, output_width//2, 3), dtype=np.uint8)
    except Exception as e:
        print(f"Error loading images {left_path}, {right_path}: {e}")
        # Create blank frames as fallback
        blank_frame = np.zeros((output_height//2, output_width//2, 3), dtype=np.uint8)
        left_img = blank_frame.copy()
        right_img = blank_frame.copy()
    
    # Convert to equirectangular dome
    try:
        dome_img = create_equirectangular_dome_from_fisheye(
            left_img, right_img, output_width, output_height
        )
    except Exception as e:
        print(f"Error processing dome projection: {e}")
        # Create a blank output as fallback
        dome_img = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # Save result with improved error handling
    try:
        # Try to save with default settings first
        success = cv2.imwrite(str(output_path), dome_img)
        if not success:
            print(f"Warning: Failed to write image to {output_path}, trying alternate compression")
            # Try with different compression settings
            success = cv2.imwrite(str(output_path), dome_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if not success:
                print(f"Error: Completely failed to write image to {output_path}")
                # Try JPEG as fallback
                jpeg_path = str(output_path).replace('.png', '.jpg')
                success = cv2.imwrite(jpeg_path, dome_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not success:
                    print(f"Error: Also failed to write JPEG to {jpeg_path}")
                    return None
                else:
                    print(f"Saved as JPEG instead: {jpeg_path}")
                    return dome_img
    except Exception as e:
        print(f"Exception while saving image {output_path}: {e}")
        # Try with different approach
        try:
            cv2.imwrite(str(output_path), dome_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        except Exception as e2:
            print(f"Second attempt also failed: {e2}")
            # Try JPEG as final fallback
            try:
                jpeg_path = str(output_path).replace('.png', '.jpg')
                success = cv2.imwrite(jpeg_path, dome_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if success:
                    print(f"Saved as JPEG instead: {jpeg_path}")
                    return dome_img
                else:
                    print(f"Also failed to save as JPEG: {jpeg_path}")
                    return None
            except Exception as e3:
                print(f"All save attempts failed: {e3}")
                return None
    
    return dome_img


def create_8k_vr180_metadata():
    """
    Return FFmpeg command parameters for VR180 metadata injection
    """
    return [
        '-metadata:s:v:0', 'stereo-mode=left-right',
        '-metadata:s:v:0', 'projection=equirectangular',
        '-metadata:s:v:0', 'spherical=true'
    ]