import numpy as np
import cv2
from pathlib import Path
from config import ULTRAFAST_MODE

def dual_fisheye_sbs_to_equirectangular_sbs(sbs_img, out_eye_w, out_eye_h):
    """
    Convert a single side-by-side dual fisheye (two circular hemispheres) frame
    to an SBS equirectangular VR180 frame (left_equi | right_equi).
    - sbs_img: input numpy BGR image with left circle in left half and right circle in right half
    - out_eye_w, out_eye_h: target per-eye equirectangular width & height (e.g., 4096 x 4096)
    Returns: equirect_sbs (out_h x 2*out_w x 3) uint8 BGR
    """
    # In ultra-fast mode, use simpler transformation for maximum speed
    if ULTRAFAST_MODE:
        # Just resize to target dimensions
        left_resized = cv2.resize(sbs_img[:, :sbs_img.shape[1]//2], (out_eye_w, out_eye_h), interpolation=cv2.INTER_LINEAR)
        right_resized = cv2.resize(sbs_img[:, sbs_img.shape[1]//2:], (out_eye_w, out_eye_h), interpolation=cv2.INTER_LINEAR)
        result = np.hstack([left_resized, right_resized])
        # Apply edge blending
        result = apply_edge_blending(result)
        return result

    h_in, w_in = sbs_img.shape[:2]
    # split left/right halves with slight overlap to avoid edge artifacts
    half_w = w_in // 2
    overlap = min(10, half_w // 20)  # Small overlap to avoid edge artifacts
    left_f = sbs_img[:, :half_w + overlap]
    right_f = sbs_img[:, half_w - overlap:]

    left_e = fisheye_to_equirectangular(left_f, out_eye_w, out_eye_h)
    right_e = fisheye_to_equirectangular(right_f, out_eye_w, out_eye_h)

    # stack side-by-side (left | right)
    out = np.hstack([left_e, right_e])
    
    # Apply immersive enhancements for "inside the scene" effect
    out = enhance_panoramic_immersion(out)
    
    # Apply edge blending for seamless panoramic effect
    out = apply_edge_blending(out)
    
    # Ensure proper data type and range
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def fisheye_to_equirectangular(fisheye_img, out_w, out_h):
    """
    Convert *one* equidistant fisheye (circular) image to equirectangular hemisphere.
    Memory-efficient implementation that processes in chunks to avoid memory issues.
    Assumptions:
      - fisheye_img contains a centered circular hemisphere (center=(cx,cy), radius=R)
      - fisheye projection is equidistant: r = f * theta  (theta = angle from centre axis)
      - we set focal f = R * 2 / pi so theta in [0, pi/2] maps to r in [0, R]
    """
    # In ultra-fast mode, use simpler transformation for maximum speed
    if ULTRAFAST_MODE:
        # Just resize to target dimensions
        return cv2.resize(fisheye_img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    
    # prepare src params
    H_in, W_in = fisheye_img.shape[:2]
    cx = W_in / 2.0
    cy = H_in / 2.0
    R = min(cx, cy) * 0.95  # Slightly reduced radius to avoid edge artifacts
    # focal length for equidistant mapping (theta in rad)
    f = 2.0 * R / np.pi  # r = f * theta, and theta in [0, pi/2] => r in [0,R]

    # Process in smaller chunks to avoid memory issues
    # Create output image
    equi = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    
    # Process in chunks of 128 rows to save memory (smaller chunks for lower resolution)
    chunk_size = min(128, out_h)
    
    for start_row in range(0, out_h, chunk_size):
        end_row = min(start_row + chunk_size, out_h)
        chunk_height = end_row - start_row
        
        # Create coordinate grids for this chunk only
        # per-eye equirectangular longitude λ ∈ [-pi/2, +pi/2] and latitude φ ∈ [+pi/2 -> -pi/2] top->bottom
        # i = x index 0..out_w-1  -> lambda = (i/out_w - 0.5) * pi
        # j = y index start_row..end_row-1  -> phi = (0.5 - (start_row + local_j)/(out_h-1)) * pi
        xs = (np.linspace(0, out_w - 1, out_w) / out_w - 0.5) * np.pi    # lambda
        local_ys = np.linspace(start_row, end_row - 1, chunk_height)
        ys = (0.5 - local_ys / (out_h - 1)) * np.pi    # phi
        lam, phi = np.meshgrid(xs, ys)   # lam,phi shape = (chunk_height, out_w)

        # 3D direction vectors for each pixel
        cos_phi = np.cos(phi)
        vx = cos_phi * np.sin(lam)   # x
        vy = np.sin(phi)             # y
        vz = cos_phi * np.cos(lam)   # z  (forward is +z)

        # theta = angle from forward axis
        # clamp numerical issues
        vz_clamped = np.clip(vz, -1.0, 1.0)
        theta = np.arccos(vz_clamped)  # 0..pi

        # For VR180 hemisphere we expect theta in [0, pi/2]
        # compute azimuth alpha = atan2(vy, vx)
        alpha = np.arctan2(vy, vx)

        # radius on fisheye image using equidistant: r = f * theta
        r = f * theta

        # source coordinates in fisheye image
        map_x = cx + r * np.cos(alpha)
        map_y = cy + r * np.sin(alpha)

        # For pixels where theta > pi/2 (outside hemisphere), map_x/y will be outside R; we'll mask later
        # Build map arrays for cv2.remap: they must be float32
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

        # Remap for this chunk only
        # Use BORDER_CONSTANT (black) for outside samples
        chunk_equi = cv2.remap(fisheye_img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

        # Mask out pixels that are outside the fisheye circle (theta > pi/2)
        # Create mask of valid sampling (r <= R * 0.999)
        valid_mask = (r <= (R * 0.999)).astype(np.uint8)  # 1 where valid
        # Set invalid areas to black
        chunk_equi[valid_mask == 0] = 0

        # Copy chunk to output image
        equi[start_row:end_row, :, :] = chunk_equi

    return equi

def enhance_panoramic_immersion(img):
    """
    Apply enhancements to create a more immersive panoramic view with "inside the scene" effect.
    Memory-efficient implementation that processes in chunks to avoid memory issues.
    This function applies:
    - Foveated emphasis (center sharp, periphery softer)
    - Subtle contrast enhancement
    - Color saturation boost
    - Light vignetting for depth perception
    """
    # In ultra-fast mode, skip enhancements for maximum speed
    if ULTRAFAST_MODE:
        return img
    
    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Create distance map from center for foveated emphasis
    # Process in chunks to save memory
    chunk_size = min(128, h)
    enhanced = img.copy()
    
    for start_row in range(0, h, chunk_size):
        end_row = min(start_row + chunk_size, h)
        chunk_height = end_row - start_row
        
        # Create coordinate grids for this chunk only
        y_chunk = np.arange(start_row, end_row)
        x_chunk = np.arange(w)
        y_grid, x_grid = np.meshgrid(y_chunk, x_chunk, indexing='ij')
        
        # Calculate distance for this chunk
        dist_chunk = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        normalized_dist_chunk = dist_chunk / max_dist
        
        # Get the image chunk
        img_chunk = img[start_row:end_row, :, :].astype(np.float32)
        enhanced_chunk = img_chunk.copy()
        
        # 1. Foveated emphasis - sharpen center, soften periphery
        center_mask = normalized_dist_chunk < 0.4  # Sharpen inner 40%
        periphery_mask = normalized_dist_chunk > 0.7  # Soften outer 30%
        
        # Sharpen center region
        if np.any(center_mask):
            kernel = np.array([[-0.5, -1, -0.5],
                               [-1, 5, -1],
                               [-0.5, -1, -0.5]]) * 0.5  # Mild sharpening
            center_sharpened = cv2.filter2D(enhanced_chunk, -1, kernel)
            # Blend sharpened center with original (stronger effect)
            enhanced_chunk[center_mask] = (
                enhanced_chunk[center_mask] * 0.7 + 
                center_sharpened[center_mask] * 0.3
            )
        
        # Soften periphery
        if np.any(periphery_mask):
            periphery_blurred = cv2.GaussianBlur(enhanced_chunk, (5, 5), 1.0)
            # Blend blurred periphery with original (subtle effect)
            enhanced_chunk[periphery_mask] = (
                enhanced_chunk[periphery_mask] * 0.8 + 
                periphery_blurred[periphery_mask] * 0.2
            )
        
        # Copy processed chunk back
        enhanced[start_row:end_row, :, :] = enhanced_chunk
    
    # Convert back to uint8 for further processing
    enhanced = enhanced.astype(np.uint8)
    
    # 2. Subtle contrast enhancement using CLAHE (process each channel separately to save memory)
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l_channel = lab[:,:,0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_channel = clahe.apply(l_channel)
    lab[:,:,0] = l_channel
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 3. Color saturation boost (subtle)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.1, 0, 255)  # Increase saturation by 10%
    hsv = hsv.astype(np.uint8)
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # 4. Light vignetting for depth perception (process in chunks)
    vignette = np.zeros((h, w), dtype=np.float32)
    for start_row in range(0, h, chunk_size):
        end_row = min(start_row + chunk_size, h)
        chunk_height = end_row - start_row
        
        # Create coordinate grids for this chunk only
        y_chunk = np.arange(start_row, end_row)
        x_chunk = np.arange(w)
        y_grid, x_grid = np.meshgrid(y_chunk, x_chunk, indexing='ij')
        
        # Calculate distance for this chunk
        dist_chunk = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        normalized_dist_chunk = dist_chunk / max_dist
        
        # Calculate vignette for this chunk
        vignette_chunk = 1.0 - (normalized_dist_chunk * 0.1)  # Max 10% darkening at edges
        vignette[start_row:end_row, :] = vignette_chunk
    
    # Apply vignette
    enhanced = enhanced.astype(np.float32)
    enhanced = enhanced * vignette[:,:,np.newaxis]
    
    # Ensure proper data type and range
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    return enhanced

def match_brightness_to_reference(output_img, ref_img):
    """Match brightness of output image to reference image"""
    # In ultra-fast mode, skip brightness matching for maximum speed
    if ULTRAFAST_MODE:
        return output_img
    
    # simple gain/gamma match using mean brightness in Y channel
    y_out = cv2.cvtColor(output_img, cv2.COLOR_BGR2YUV)[:,:,0].astype(np.float32)
    y_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2YUV)[:,:,0].astype(np.float32)
    gain = (y_ref.mean() + 1e-8) / (y_out.mean() + 1e-8)
    out = np.clip(output_img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return out

def fix_color_artifacts(img):
    """Fix color artifacts in the output image"""
    # In ultra-fast mode, skip complex filtering for maximum speed
    if ULTRAFAST_MODE:
        return img
    
    # Apply bilateral filter to reduce color artifacts while preserving edges
    filtered = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Apply slight Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(filtered, (3, 3), 0)
    
    # Ensure proper data type and range
    result = np.clip(blurred, 0, 255).astype(np.uint8)
    return result

def align_stereo_views(left_img, right_img):
    """Align left and right eye views for proper stereo effect"""
    # In ultra-fast mode, skip alignment for maximum speed
    if ULTRAFAST_MODE:
        return left_img, right_img
    
    # Convert to grayscale for feature detection
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    # Detect features using ORB
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(left_gray, None)
    kp2, des2 = orb.detectAndCompute(right_gray, None)
    
    # Match features
    if des1 is not None and des2 is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Extract matched points
        if len(matches) > 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # Find homography matrix
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            # Apply homography to align right image to left
            if M is not None:
                h, w = left_img.shape[:2]
                right_aligned = cv2.warpPerspective(right_img, M, (w, h))
                return left_img, right_aligned
    
    # If alignment fails, return original images
    return left_img, right_img

def apply_edge_blending(img):
    """
    Apply edge blending to create seamless panoramic effect
    This helps avoid visible borders after projection
    """
    h, w = img.shape[:2]
    
    # Create a mask for edge blending
    # Blend the edges of left and right eye images
    blend_width = max(50, w // 32)  # Minimum 50 pixels, up to 3.125% of width for blending
    
    # Create gradient mask for blending
    mask = np.ones((h, w), dtype=np.float32)
    
    # Blend left edge of right eye (right half of image)
    right_eye_start = w // 2
    right_eye_end = min(right_eye_start + blend_width, w)
    
    if right_eye_end > right_eye_start:
        # Create smooth cosine gradient from 1.0 to 0.0
        gradient = np.cos(np.linspace(0, np.pi/2, right_eye_end - right_eye_start)) ** 2
        mask[:, right_eye_start:right_eye_end] = np.tile(gradient, (h, 1))
    
    # Also blend right edge of left eye (left half of image)
    left_eye_end = w // 2
    left_eye_start = max(0, left_eye_end - blend_width)
    
    if left_eye_end > left_eye_start:
        # Create smooth cosine gradient from 0.0 to 1.0
        gradient = np.cos(np.linspace(np.pi/2, 0, left_eye_end - left_eye_start)) ** 2
        mask[:, left_eye_start:left_eye_end] = np.tile(gradient, (h, 1))
    
    # Apply Gaussian blur to the mask for smoother transition
    # Create a 1D kernel for horizontal blurring only
    blur_kernel_size = max(5, blend_width // 10)
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1  # Ensure odd kernel size
    
    # Apply horizontal blur only to preserve vertical details
    mask_blurred = cv2.GaussianBlur(mask, (blur_kernel_size, 1), 0)
    
    # Apply the mask to create blended result
    # Use the mask to blend with a slightly blurred version for smooth transitions
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Blend original and blurred based on mask
    mask_3d = np.stack([mask_blurred, mask_blurred, mask_blurred], axis=2)
    result = (img * mask_3d + blurred * (1 - mask_3d)).astype(np.uint8)
    
    return result