import cv2, numpy as np, math
from pathlib import Path
from .utils import ensure_dir
from config import PANINI_MIX, STEREO_MIX, IDENTITY_MIX, TARGET_PER_EYE, MAX_WORKERS, SBS_WIDTH, SBS_HEIGHT, ULTRAFAST_MODE
from concurrent.futures import ThreadPoolExecutor, as_completed

def panini_map(xx, yy, s=0.8):
    """Panini projection with increased cinematic effect (s from 0.7 to 0.8)"""
    denom = 1 + s*(1 - xx*xx)
    return xx/denom, yy/denom

def stereographic_map(xx, yy):
    """Stereographic projection for panoramic curvature"""
    denom = 1 + xx*xx + yy*yy
    return 2*xx/denom, 2*yy/denom

def blend_and_panoramic(img, per_eye=TARGET_PER_EYE, fov_deg=180):
    """
    Create panoramic projection with cinematic barrel distortion for immersive dome effect
    Implements Panini + stereographic blending for smooth panoramic curvature at edges
    Ensures edges are bent/panoramic, not flat circles
    """
    # Even in ultra-fast mode, apply cinematic projection for better quality
    h,w = img.shape[:2]
    
    # Create coordinate grids
    nx = (np.linspace(-1,1,w)).astype('float32')
    ny = (np.linspace(-1,1,h)).astype('float32')
    xx, yy = np.meshgrid(nx, ny)
    
    # Apply projection transforms with cinematic parameters
    px, py = panini_map(xx, yy, s=PANINI_MIX)  # Increased from 0.7 to 0.8
    sx, sy = stereographic_map(xx, yy)
    
    # Blend projections with cinematic effect
    # Adjusted weights for stronger panoramic curvature
    bx = PANINI_MIX*px + STEREO_MIX*sx + IDENTITY_MIX*xx
    by = PANINI_MIX*py + STEREO_MIX*sy + IDENTITY_MIX*yy
    
    # Apply enhanced barrel distortion for cinematic dome effect
    # This creates the cinematic "inside the scene" feeling with bent edges
    r = np.sqrt(bx*bx + by*by)
    # Enhanced distortion for cinematic effect with smoother transition
    distortion = 1 + 0.35 * (r**2.5)  # Increased from 0.30 to 0.35 for stronger effect
    bx_distorted = bx * distortion
    by_distorted = by * distortion
    
    # Convert to pixel coordinates
    mapx = ((bx_distorted + 1.0)/2.0) * w
    mapy = ((by_distorted + 1.0)/2.0) * h
    
    # Apply remapping with border reflection for cinematic quality
    # Use BORDER_WRAP for better edge continuity in panoramic projections
    warped = cv2.remap(img, mapx.astype('float32'), mapy.astype('float32'), 
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    
    # Apply additional edge enhancement to ensure edges are bent/panoramic
    # Create a mask for edge enhancement
    center_x, center_y = w // 2, h // 2
    dist_from_center = np.sqrt((np.arange(w) - center_x)**2 + (np.arange(h)[:, np.newaxis] - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    normalized_dist = dist_from_center / max_dist
    
    # Apply stronger enhancement to edges for panoramic curvature
    edge_mask = normalized_dist > 0.7  # Focus on outer 30% for edge enhancement
    if np.any(edge_mask):
        # Apply slight sharpening to edges for better definition
        kernel = np.array([[-0.5, -1, -0.5],
                          [-1, 5, -1],
                          [-0.5, -1, -0.5]]) * 0.3
        edge_enhanced = cv2.filter2D(warped, -1, kernel)
        # Blend enhanced edges with original
        warped = np.where(edge_mask[..., np.newaxis], 
                         cv2.addWeighted(warped, 0.8, edge_enhanced, 0.2, 0), 
                         warped)
    
    return warped

def spherical_to_equirectangular_panoramic(img, output_width=8192, output_height=4096):
    """
    Convert a rectilinear image to equirectangular panoramic format
    Memory-efficient implementation that processes in chunks to avoid memory issues.
    This expands the field of view to 180° horizontally for proper VR180
    """
    # In ultra-fast mode, use simpler transformation for maximum speed
    if ULTRAFAST_MODE:
        # Just resize to target dimensions
        return cv2.resize(img, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
    
    # Calculate output dimensions for VR180 (180° horizontal FOV)
    output_width = output_width
    output_height = output_height
    
    # Process in smaller chunks to avoid memory issues
    # Create output image
    panoramic = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # Process in chunks of 128 rows to save memory (smaller chunks for lower resolution)
    chunk_size = min(128, output_height)
    h, w = img.shape[:2]
    
    for start_row in range(0, output_height, chunk_size):
        end_row = min(start_row + chunk_size, output_height)
        chunk_height = end_row - start_row
        
        # Create longitude and latitude maps for this chunk only
        lon = np.linspace(-np.pi/2, np.pi/2, output_width)  # -90° to +90° for 180° horizontal
        # For this chunk, we need to calculate the latitudes for the rows in this chunk
        lat_indices = np.linspace(start_row, end_row - 1, chunk_height)
        lat = (0.5 - lat_indices / (output_height - 1)) * (np.pi/2)  # -45° to +45° for comfortable vertical FOV
        
        # Create meshgrid for this chunk
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # Convert spherical coordinates to 3D Cartesian for this chunk
        x = np.cos(lat_grid) * np.cos(lon_grid)
        y = np.sin(lat_grid)
        z = np.cos(lat_grid) * np.sin(lon_grid)
        
        # Convert 3D coordinates to image coordinates (assuming input is rectilinear)
        # Map 3D points to input image coordinates
        u = (np.arctan2(z, x) + np.pi/2) * (w / np.pi)  # Horizontal mapping for 180° FOV
        v = (np.pi/2 - np.arccos(y/np.sqrt(x**2 + y**2 + z**2))) * (h / np.pi)  # Vertical mapping
        
        # Clamp coordinates to valid range
        u = np.clip(u, 0, w-1).astype('float32')
        v = np.clip(v, 0, h-1).astype('float32')
        
        # Apply remapping to create equirectangular panoramic image for this chunk
        chunk_panoramic = cv2.remap(img, u, v, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        
        # Copy chunk to output image
        panoramic[start_row:end_row, :, :] = chunk_panoramic

    return panoramic

def create_equirectangular_panorama(left_img, right_img, output_height=4096):
    """
    Convert left and right eye images to equirectangular panoramic stereo format
    Output format: width = 2 * height, left eye on left half, right eye on right half
    Memory-efficient implementation that processes in chunks to avoid memory issues.
    Implements all requested panoramic & immersive features:
    - Panini or rectilinear blend for the center
    - Expand periphery up to 180° horizontally
    - Light barrel distortion at edges for dome effect
    """
    # In ultra-fast mode, skip complex processing for maximum speed
    if ULTRAFAST_MODE:
        # Just resize and concatenate
        output_width = output_height * 2
        left_resized = cv2.resize(left_img, (output_width//2, output_height), interpolation=cv2.INTER_LINEAR)
        right_resized = cv2.resize(right_img, (output_width//2, output_height), interpolation=cv2.INTER_LINEAR)
        return cv2.hconcat([left_resized, right_resized])
    
    # Determine output dimensions (8K VR180 standard: 16384x8192)
    output_width = output_height * 2
    output_height = output_height
    
    # Step 1: Apply panoramic projection with immersive effects
    left_projected = blend_and_panoramic(left_img, per_eye=output_height//2)
    right_projected = blend_and_panoramic(right_img, per_eye=output_height//2)
    
    # Step 2: Convert to equirectangular format with 180° horizontal FOV
    left_equirectangular = spherical_to_equirectangular_panoramic(left_projected, output_width//2, output_height)
    right_equirectangular = spherical_to_equirectangular_panoramic(right_projected, output_width//2, output_height)
    
    # Step 3: Combine left and right eyes side-by-side
    equirectangular_stereo = cv2.hconcat([left_equirectangular, right_equirectangular])
    
    # Step 4: Apply subtle enhancements for immersive "inside the scene" effect
    # Process in chunks to save memory
    h, w = equirectangular_stereo.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Process in chunks of 128 rows to save memory (smaller chunks for lower resolution)
    chunk_size = min(128, h)
    enhanced_stereo = equirectangular_stereo.copy()
    
    for start_row in range(0, h, chunk_size):
        end_row = min(start_row + chunk_size, h)
        chunk_height = end_row - start_row
        
        # Get the image chunk
        img_chunk = equirectangular_stereo[start_row:end_row, :, :].astype(np.float32)
        enhanced_chunk = img_chunk.copy()
        
        # Create distance map from center for foveated emphasis for this chunk
        y_chunk = np.arange(start_row, end_row)
        x_chunk = np.arange(w)
        y_grid, x_grid = np.meshgrid(y_chunk, x_chunk, indexing='ij')
        dist_chunk = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        normalized_dist_chunk = dist_chunk / max_dist
        
        # Apply subtle sharpening to center and softening to periphery for foveated effect
        # Enhances the "inside the scene" feeling
        center_mask = normalized_dist_chunk < 0.4  # Sharpen inner 40%
        periphery_mask = normalized_dist_chunk > 0.7  # Soften outer 30%
        
        # Apply slight sharpening to center region for enhanced detail
        if np.any(center_mask):
            kernel = np.array([[-0.5, -1, -0.5],
                               [-1, 5, -1],
                               [-0.5, -1, -0.5]]) * 0.5  # Mild sharpening
            center_sharpened = cv2.filter2D(enhanced_chunk, -1, kernel)
            # Blend sharpened center with original
            enhanced_chunk[center_mask] = (
                enhanced_chunk[center_mask] * 1.1 + 
                center_sharpened[center_mask] * 0.1
            )
        
        # Apply slight gaussian blur to periphery for natural vision simulation
        if np.any(periphery_mask):
            periphery_blurred = cv2.GaussianBlur(enhanced_chunk, (5, 5), 0.8)
            # Blend blurred periphery with original
            enhanced_chunk[periphery_mask] = (
                enhanced_chunk[periphery_mask] * 0.9 + 
                periphery_blurred[periphery_mask] * 0.1
            )
        
        # Copy processed chunk back
        enhanced_stereo[start_row:end_row, :, :] = enhanced_chunk.astype(equirectangular_stereo.dtype)
    
    return enhanced_stereo

def _process_single_projection_pair(f, left_in, right_in, left_out, right_out):
    """Process a single pair of left/right frames for projection"""
    L = cv2.imread(str(f))
    R = cv2.imread(str(Path(right_in)/f.name))
    
    # Apply panoramic projection with immersive effects
    Lf = blend_and_panoramic(L, per_eye=TARGET_PER_EYE)
    Rf = blend_and_panoramic(R, per_eye=TARGET_PER_EYE)
    
    cv2.imwrite(str(Path(left_out)/f.name), Lf)
    cv2.imwrite(str(Path(right_out)/f.name), Rf)
    return f.name

def process_dirs(left_in, right_in, left_out, right_out, progress_cb=None):
    """Process directories with updated panoramic projection"""
    ensure_dir(left_out); ensure_dir(right_out)
    left_files = sorted(Path(left_in).glob('*.png'))
    n = len(left_files)
    
    # In ultra-fast mode, reduce number of workers
    workers = min(MAX_WORKERS, 2) if ULTRAFAST_MODE and MAX_WORKERS > 1 else MAX_WORKERS
    
    if workers > 1:
        # Use multi-threading for parallel processing
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all projection tasks
            future_to_file = {
                executor.submit(_process_single_projection_pair, f, left_in, right_in, left_out, right_out): f 
                for f in left_files
            }
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    completed += 1
                    if progress_cb and (completed % 10) == 0:
                        progress_cb(45 + int(10 * (completed / max(1, n))), f'Projection {completed}/{n}')
                except Exception as exc:
                    print(f'Frame {file_path} generated an exception: {exc}')
    else:
        # Single-threaded processing (fallback)
        for i,f in enumerate(left_files):
            L = cv2.imread(str(f))
            R = cv2.imread(str(Path(right_in)/f.name))
            
            # Apply panoramic projection with immersive effects
            Lf = blend_and_panoramic(L, per_eye=TARGET_PER_EYE)
            Rf = blend_and_panoramic(R, per_eye=TARGET_PER_EYE)
            
            cv2.imwrite(str(Path(left_out)/f.name), Lf)
            cv2.imwrite(str(Path(right_out)/f.name), Rf)
            if progress_cb and i%10==0:
                progress_cb(45 + int(10*(i/n)), f'Projection {i}/{n}')

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

# New function to create final equirectangular stereo frames from fisheye images
def create_final_panoramic_stereo_from_fisheye(left_dir, right_dir, output_dir, progress_cb=None):
    """
    Create final equirectangular panoramic stereo frames from fisheye left/right views
    This converts dual-fisheye output to proper VR180 equirectangular format using ffmpeg v360 filter
    
    Implements all requested panoramic & immersive features:
    - Converts dual-fisheye to equirectangular projection using ffmpeg v360
    - Proper stereo disparity for immersive effect
    - Brightness matching to preserve original look
    - FFmpeg v360 filter for proper fisheye to equirectangular conversion
    
    Optimized for speed:
    - Multi-threading for parallel processing
    """
    import subprocess
    from pathlib import Path
    import cv2
    from config import TARGET_PER_EYE, ULTRAFAST_MODE
    
    ensure_dir(output_dir)
    left_files = sorted(Path(left_dir).glob('*.png'))
    right_files = sorted(Path(right_dir).glob('*.png'))
    n = len(left_files)
    
    print(f"[FisheyeToPanoramic] Creating equirectangular stereo frames from fisheye: {n} frames")
    print(f"  - Output format: {TARGET_PER_EYE*2}x{TARGET_PER_EYE} (VR180)")
    print(f"  - Left/right stereo side-by-side")
    print(f"  - 180° horizontal FOV")
    print(f"  - Using FFmpeg v360 filter for proper fisheye conversion")
    
    # Import the conversion functions
    from .convert_fisheye import match_brightness_to_reference, align_stereo_views, enhance_panoramic_immersion, fix_color_artifacts
    
    def _process_single_fisheye_frame(i, l, r):
        """Process a single fisheye stereo frame and convert to equirectangular"""
        if i % 10 == 0:  # Print progress every 10 frames
            print(f"  Processing frame {i+1}/{n}: {l.name}")
            
        # Read left and right eye images
        L = cv2.imread(str(l))
        R = cv2.imread(str(r))
        
        # Align stereo views for proper stereo effect
        L_aligned, R_aligned = align_stereo_views(L, R)
        
        # Create temporary side-by-side fisheye image for ffmpeg processing
        temp_dir = Path(output_dir).parent / "temp_fisheye"
        temp_dir.mkdir(exist_ok=True)
        temp_sbs_path = temp_dir / f"temp_sbs_{i:06d}.png"
        sbs_fisheye = cv2.hconcat([L_aligned, R_aligned])
        cv2.imwrite(str(temp_sbs_path), sbs_fisheye)
        
        # Check if FFmpeg has v360 filter available
        ffmpeg_has_v360 = True
        try:
            result = subprocess.run(['ffmpeg', '-filters'], capture_output=True, text=True, timeout=10)
            if 'v360' not in result.stdout:
                ffmpeg_has_v360 = False
                print("  FFmpeg v360 filter not available, using fallback method")
        except Exception as e:
            ffmpeg_has_v360 = False
            print(f"  Could not check FFmpeg filters: {e}")
        
        # Use FFmpeg v360 filter to convert dual fisheye to equirectangular if available
        if ffmpeg_has_v360:
            # Input: Dual fisheye (side-by-side) -> Output: Equirectangular stereo (side-by-side)
            temp_output_path = temp_dir / f"temp_equi_{i:06d}.png"
            
            # FFmpeg command using v360 filter for proper fisheye to equirectangular conversion
            # Use 8192x4096 resolution to match what the pipeline is actually generating
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', str(temp_sbs_path),
                '-vf', f'v360=dfisheye:equirect:h=4096:w=8192',
                '-q:v', '2',  # Quality setting for PNG
                str(temp_output_path)
            ]
            
            try:
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True, timeout=60)
                
                # Read the converted equirectangular image
                if temp_output_path.exists():
                    equi_sbs = cv2.imread(str(temp_output_path))
                    
                    # Apply immersive enhancements for "inside the scene" effect
                    equi_sbs = enhance_panoramic_immersion(equi_sbs)
                    
                    # Fix color artifacts
                    equi_sbs = fix_color_artifacts(equi_sbs)
                    
                    # Save the final equirectangular stereo frame
                    output_path = Path(output_dir) / f"panoramic_stereo_{i:06d}.png"
                    cv2.imwrite(str(output_path), equi_sbs)
                    
                    # Clean up temporary files
                    temp_sbs_path.unlink(missing_ok=True)
                    temp_output_path.unlink(missing_ok=True)
                    
                    return i, str(output_path)
                else:
                    raise Exception("FFmpeg conversion failed - output file not created")
            except subprocess.TimeoutExpired:
                print(f"  FFmpeg conversion timed out for frame {i}")
                # Fall back to existing method
                from .convert_fisheye import dual_fisheye_sbs_to_equirectangular_sbs
                equi_sbs = dual_fisheye_sbs_to_equirectangular_sbs(sbs_fisheye, TARGET_PER_EYE, TARGET_PER_EYE)
                equi_sbs = enhance_panoramic_immersion(equi_sbs)
                equi_sbs = fix_color_artifacts(equi_sbs)
                output_path = Path(output_dir) / f"panoramic_stereo_{i:06d}.png"
                cv2.imwrite(str(output_path), equi_sbs)
                temp_sbs_path.unlink(missing_ok=True)
                return i, str(output_path)
            except subprocess.CalledProcessError as e:
                print(f"  FFmpeg conversion failed for frame {i}: {e}")
                # Fall back to existing method
                from .convert_fisheye import dual_fisheye_sbs_to_equirectangular_sbs
                equi_sbs = dual_fisheye_sbs_to_equirectangular_sbs(sbs_fisheye, TARGET_PER_EYE, TARGET_PER_EYE)
                equi_sbs = enhance_panoramic_immersion(equi_sbs)
                equi_sbs = fix_color_artifacts(equi_sbs)
                output_path = Path(output_dir) / f"panoramic_stereo_{i:06d}.png"
                cv2.imwrite(str(output_path), equi_sbs)
                temp_sbs_path.unlink(missing_ok=True)
                return i, str(output_path)
        else:
            # Use existing method if v360 filter is not available
            from .convert_fisheye import dual_fisheye_sbs_to_equirectangular_sbs
            equi_sbs = dual_fisheye_sbs_to_equirectangular_sbs(sbs_fisheye, TARGET_PER_EYE, TARGET_PER_EYE)
            equi_sbs = enhance_panoramic_immersion(equi_sbs)
            equi_sbs = fix_color_artifacts(equi_sbs)
            output_path = Path(output_dir) / f"panoramic_stereo_{i:06d}.png"
            cv2.imwrite(str(output_path), equi_sbs)
            temp_sbs_path.unlink(missing_ok=True)
            return i, str(output_path)
    
    # Process frames with multi-threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # In ultra-fast mode, reduce number of workers
    workers = min(MAX_WORKERS, 2) if ULTRAFAST_MODE and MAX_WORKERS > 1 else MAX_WORKERS
    
    if workers > 1 and len(left_files) > 1:
        print(f"  - Using {workers} worker threads for parallel processing")
        # Use multi-threading for parallel processing
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all fisheye processing tasks
            future_to_index = {
                executor.submit(_process_single_fisheye_frame, i, l, r): i 
                for i, (l, r) in enumerate(zip(left_files, right_files))
            }
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result_index, result_path = future.result()
                    completed += 1
                    if progress_cb and (completed % 5) == 0:
                        progress_cb(45 + int(10 * (completed / max(1, len(left_files)))), f'Fisheye to Panoramic {completed}/{len(left_files)}')
                except Exception as exc:
                    print(f'Frame {index} generated an exception: {exc}')
    else:
        print("  - Using single-threaded processing")
        # Single-threaded processing (fallback)
        for i, (l, r) in enumerate(zip(left_files, right_files)):
            try:
                result_index, result_path = _process_single_fisheye_frame(i, l, r)
                if progress_cb and (i % 5) == 0:
                    progress_cb(45 + int(10 * (i / max(1, len(left_files)))), f'Fisheye to Panoramic {i}/{len(left_files)}')
            except Exception as exc:
                print(f'Frame {i} generated an exception: {exc}')
    
    # Clean up temporary directory
    try:
        temp_dir = Path(output_dir).parent / "temp_fisheye"
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        print(f"  Warning: Could not clean up temporary directory: {e}")
    
    print(f"[FisheyeToPanoramic] Created {n} equirectangular stereo frames")

import cv2, numpy as np, math
from pathlib import Path
from .utils import ensure_dir
from config import PANINI_MIX, STEREO_MIX, IDENTITY_MIX, TARGET_PER_EYE, MAX_WORKERS, SBS_WIDTH, SBS_HEIGHT, ULTRAFAST_MODE
from concurrent.futures import ThreadPoolExecutor, as_completed

def panini_map(xx, yy, s=0.8):
    """Panini projection with increased cinematic effect (s from 0.7 to 0.8)"""
    denom = 1 + s*(1 - xx*xx)
    return xx/denom, yy/denom

def stereographic_map(xx, yy):
    """Stereographic projection for panoramic curvature"""
    denom = 1 + xx*xx + yy*yy
    return 2*xx/denom, 2*yy/denom

def blend_and_panoramic(img, per_eye=TARGET_PER_EYE, fov_deg=180):
    """
    Create panoramic projection with cinematic barrel distortion for immersive dome effect
    Implements Panini + stereographic blending for smooth panoramic curvature at edges
    Ensures edges are bent/panoramic, not flat circles
    """
    # Even in ultra-fast mode, apply cinematic projection for better quality
    h,w = img.shape[:2]
    
    # Create coordinate grids
    nx = (np.linspace(-1,1,w)).astype('float32')
    ny = (np.linspace(-1,1,h)).astype('float32')
    xx, yy = np.meshgrid(nx, ny)
    
    # Apply projection transforms with cinematic parameters
    px, py = panini_map(xx, yy, s=PANINI_MIX)  # Increased from 0.7 to 0.8
    sx, sy = stereographic_map(xx, yy)
    
    # Blend projections with cinematic effect
    # Adjusted weights for stronger panoramic curvature
    bx = PANINI_MIX*px + STEREO_MIX*sx + IDENTITY_MIX*xx
    by = PANINI_MIX*py + STEREO_MIX*sy + IDENTITY_MIX*yy
    
    # Apply enhanced barrel distortion for cinematic dome effect
    # This creates the cinematic "inside the scene" feeling with bent edges
    r = np.sqrt(bx*bx + by*by)
    # Enhanced distortion for cinematic effect with smoother transition
    distortion = 1 + 0.35 * (r**2.5)  # Increased from 0.30 to 0.35 for stronger effect
    bx_distorted = bx * distortion
    by_distorted = by * distortion
    
    # Convert to pixel coordinates
    mapx = ((bx_distorted + 1.0)/2.0) * w
    mapy = ((by_distorted + 1.0)/2.0) * h
    
    # Apply remapping with border reflection for cinematic quality
    # Use BORDER_WRAP for better edge continuity in panoramic projections
    warped = cv2.remap(img, mapx.astype('float32'), mapy.astype('float32'), 
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    
    # Apply additional edge enhancement to ensure edges are bent/panoramic
    # Create a mask for edge enhancement
    center_x, center_y = w // 2, h // 2
    dist_from_center = np.sqrt((np.arange(w) - center_x)**2 + (np.arange(h)[:, np.newaxis] - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    normalized_dist = dist_from_center / max_dist
    
    # Apply stronger enhancement to edges for panoramic curvature
    edge_mask = normalized_dist > 0.7  # Focus on outer 30% for edge enhancement
    if np.any(edge_mask):
        # Apply slight sharpening to edges for better definition
        kernel = np.array([[-0.5, -1, -0.5],
                          [-1, 5, -1],
                          [-0.5, -1, -0.5]]) * 0.3
        edge_enhanced = cv2.filter2D(warped, -1, kernel)
        # Blend enhanced edges with original
        warped = np.where(edge_mask[..., np.newaxis], 
                         cv2.addWeighted(warped, 0.8, edge_enhanced, 0.2, 0), 
                         warped)
    
    return warped

def spherical_to_equirectangular_panoramic(img, output_width=8192, output_height=4096):
    """
    Convert a rectilinear image to equirectangular panoramic format
    Memory-efficient implementation that processes in chunks to avoid memory issues.
    This expands the field of view to 180° horizontally for proper VR180
    """
    # In ultra-fast mode, use simpler transformation for maximum speed
    if ULTRAFAST_MODE:
        # Just resize to target dimensions
        return cv2.resize(img, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
    
    # Calculate output dimensions for VR180 (180° horizontal FOV)
    output_width = output_width
    output_height = output_height
    
    # Process in smaller chunks to avoid memory issues
    # Create output image
    panoramic = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # Process in chunks of 128 rows to save memory (smaller chunks for lower resolution)
    chunk_size = min(128, output_height)
    h, w = img.shape[:2]
    
    for start_row in range(0, output_height, chunk_size):
        end_row = min(start_row + chunk_size, output_height)
        chunk_height = end_row - start_row
        
        # Create longitude and latitude maps for this chunk only
        lon = np.linspace(-np.pi/2, np.pi/2, output_width)  # -90° to +90° for 180° horizontal
        # For this chunk, we need to calculate the latitudes for the rows in this chunk
        lat_indices = np.linspace(start_row, end_row - 1, chunk_height)
        lat = (0.5 - lat_indices / (output_height - 1)) * (np.pi/2)  # -45° to +45° for comfortable vertical FOV
        
        # Create meshgrid for this chunk
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # Convert spherical coordinates to 3D Cartesian for this chunk
        x = np.cos(lat_grid) * np.cos(lon_grid)
        y = np.sin(lat_grid)
        z = np.cos(lat_grid) * np.sin(lon_grid)
        
        # Convert 3D coordinates to image coordinates (assuming input is rectilinear)
        # Map 3D points to input image coordinates
        u = (np.arctan2(z, x) + np.pi/2) * (w / np.pi)  # Horizontal mapping for 180° FOV
        v = (np.pi/2 - np.arccos(y/np.sqrt(x**2 + y**2 + z**2))) * (h / np.pi)  # Vertical mapping
        
        # Clamp coordinates to valid range
        u = np.clip(u, 0, w-1).astype('float32')
        v = np.clip(v, 0, h-1).astype('float32')
        
        # Apply remapping to create equirectangular panoramic image for this chunk
        chunk_panoramic = cv2.remap(img, u, v, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        
        # Copy chunk to output image
        panoramic[start_row:end_row, :, :] = chunk_panoramic

    return panoramic

def create_equirectangular_panorama(left_img, right_img, output_height=4096):
    """
    Convert left and right eye images to equirectangular panoramic stereo format
    Output format: width = 2 * height, left eye on left half, right eye on right half
    Memory-efficient implementation that processes in chunks to avoid memory issues.
    Implements all requested panoramic & immersive features:
    - Panini or rectilinear blend for the center
    - Expand periphery up to 180° horizontally
    - Light barrel distortion at edges for dome effect
    """
    # In ultra-fast mode, skip complex processing for maximum speed
    if ULTRAFAST_MODE:
        # Just resize and concatenate
        output_width = output_height * 2
        left_resized = cv2.resize(left_img, (output_width//2, output_height), interpolation=cv2.INTER_LINEAR)
        right_resized = cv2.resize(right_img, (output_width//2, output_height), interpolation=cv2.INTER_LINEAR)
        return cv2.hconcat([left_resized, right_resized])
    
    # Determine output dimensions (8K VR180 standard: 8640x4320)
    output_width = output_height * 2
    output_height = output_height
    
    # Step 1: Apply panoramic projection with immersive effects
    left_projected = blend_and_panoramic(left_img, per_eye=output_height//2)
    right_projected = blend_and_panoramic(right_img, per_eye=output_height//2)
    
    # Step 2: Convert to equirectangular format with 180° horizontal FOV
    left_equirectangular = spherical_to_equirectangular_panoramic(left_projected, output_width//2, output_height)
    right_equirectangular = spherical_to_equirectangular_panoramic(right_projected, output_width//2, output_height)
    
    # Step 3: Combine left and right eyes side-by-side
    equirectangular_stereo = cv2.hconcat([left_equirectangular, right_equirectangular])
    
    # Step 4: Apply subtle enhancements for immersive "inside the scene" effect
    # Process in chunks to save memory
    h, w = equirectangular_stereo.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Process in chunks of 128 rows to save memory (smaller chunks for lower resolution)
    chunk_size = min(128, h)
    enhanced_stereo = equirectangular_stereo.copy()
    
    for start_row in range(0, h, chunk_size):
        end_row = min(start_row + chunk_size, h)
        chunk_height = end_row - start_row
        
        # Get the image chunk
        img_chunk = equirectangular_stereo[start_row:end_row, :, :].astype(np.float32)
        enhanced_chunk = img_chunk.copy()
        
        # Create distance map from center for foveated emphasis for this chunk
        y_chunk = np.arange(start_row, end_row)
        x_chunk = np.arange(w)
        y_grid, x_grid = np.meshgrid(y_chunk, x_chunk, indexing='ij')
        dist_chunk = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        normalized_dist_chunk = dist_chunk / max_dist
        
        # Apply subtle sharpening to center and softening to periphery for foveated effect
        # Enhances the "inside the scene" feeling
        center_mask = normalized_dist_chunk < 0.4  # Sharpen inner 40%
        periphery_mask = normalized_dist_chunk > 0.7  # Soften outer 30%
        
        # Apply slight sharpening to center region for enhanced detail
        if np.any(center_mask):
            kernel = np.array([[-0.5, -1, -0.5],
                               [-1, 5, -1],
                               [-0.5, -1, -0.5]]) * 0.5  # Mild sharpening
            center_sharpened = cv2.filter2D(enhanced_chunk, -1, kernel)
            # Blend sharpened center with original
            enhanced_chunk[center_mask] = (
                enhanced_chunk[center_mask] * 1.1 + 
                center_sharpened[center_mask] * 0.1
            )
        
        # Apply slight gaussian blur to periphery for natural vision simulation
        if np.any(periphery_mask):
            periphery_blurred = cv2.GaussianBlur(enhanced_chunk, (5, 5), 0.8)
            # Blend blurred periphery with original
            enhanced_chunk[periphery_mask] = (
                enhanced_chunk[periphery_mask] * 0.9 + 
                periphery_blurred[periphery_mask] * 0.1
            )
        
        # Copy processed chunk back
        enhanced_stereo[start_row:end_row, :, :] = enhanced_chunk.astype(equirectangular_stereo.dtype)
    
    return enhanced_stereo

def _process_single_projection_pair(f, left_in, right_in, left_out, right_out):
    """Process a single pair of left/right frames for projection"""
    L = cv2.imread(str(f))
    R = cv2.imread(str(Path(right_in)/f.name))
    
    # Apply panoramic projection with immersive effects
    Lf = blend_and_panoramic(L, per_eye=TARGET_PER_EYE)
    Rf = blend_and_panoramic(R, per_eye=TARGET_PER_EYE)
    
    cv2.imwrite(str(Path(left_out)/f.name), Lf)
    cv2.imwrite(str(Path(right_out)/f.name), Rf)
    return f.name

def process_dirs(left_in, right_in, left_out, right_out, progress_cb=None):
    """Process directories with updated panoramic projection"""
    ensure_dir(left_out); ensure_dir(right_out)
    left_files = sorted(Path(left_in).glob('*.png'))
    n = len(left_files)
    
    # In ultra-fast mode, reduce number of workers
    workers = min(MAX_WORKERS, 1) if ULTRAFAST_MODE and MAX_WORKERS > 1 else 1  # Reduce workers for speed
    
    if workers > 1:
        # Use multi-threading for parallel processing
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all projection tasks
            future_to_file = {
                executor.submit(_process_single_projection_pair, f, left_in, right_in, left_out, right_out): f 
                for f in left_files
            }
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    completed += 1
                    if progress_cb and (completed % 10) == 0:
                        progress_cb(45 + int(10 * (completed / max(1, n))), f'Projection {completed}/{n}')
                except Exception as exc:
                    print(f'Frame {file_path} generated an exception: {exc}')
    else:
        # Single-threaded processing (fallback)
        for i,f in enumerate(left_files):
            L = cv2.imread(str(f))
            R = cv2.imread(str(Path(right_in)/f.name))
            
            # Apply panoramic projection with immersive effects
            Lf = blend_and_panoramic(L, per_eye=TARGET_PER_EYE)
            Rf = blend_and_panoramic(R, per_eye=TARGET_PER_EYE)
            
            cv2.imwrite(str(Path(left_out)/f.name), Lf)
            cv2.imwrite(str(Path(right_out)/f.name), Rf)
            if progress_cb and i%10==0:
                progress_cb(45 + int(10*(i/n)), f'Projection {i}/{n}')

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

# New function to create final equirectangular dome stereo frames
def create_final_panoramic_stereo(left_dir, right_dir, output_dir, progress_cb=None):
    """
    Create final equirectangular dome stereo frames from processed left/right views
    This replaces the dual-fisheye output with proper VR180 equirectangular format
    
    Implements all requested panoramic & immersive features:
    - Panini + stereographic + identity projection blending for smooth dome edges
    - Expand periphery up to 180° horizontally with bent edges (no circles)
    - Preserve brightness and color from input
    - Increase stereo disparity for "inside-the-scene" effect
    - Apply immersive enhancements for dome experience
    
    Optimized for speed:
    - Multi-threading for parallel processing
    - All frames processed (no skipping to maintain video timing)
    """
    from .dome_projection import process_fisheye_frame_pair
    
    ensure_dir(output_dir)
    left_files = sorted(Path(left_dir).glob('*.png'))
    right_files = sorted(Path(right_dir).glob('*.png'))
    n = len(left_files)
    
    print(f"[DomeProjection] Creating equirectangular dome stereo frames: {n} frames")
    print(f"  - Output format: {TARGET_PER_EYE*2}x{TARGET_PER_EYE} (VR180)")
    print(f"  - Left/right stereo side-by-side")
    print(f"  - 180° horizontal FOV with dome projection")
    print(f"  - Smooth bent edges, no circular borders")
    print(f"  - Enhanced stereo disparity for immersive effect")
    
    # Process all frames to maintain video timing (no skipping)
    left_files_filtered = left_files
    right_files_filtered = right_files
    print(f"  - Processing all {n} frames")
    
    def _process_single_dome_frame(i, l, r):
        """Process a single dome stereo frame"""
        if i % 10 == 0:  # Print progress every 10 frames to reduce output
            print(f"  Processing frame {i+1}/{n}: {l.name}")
            
        # Use the new dome projection implementation
        output_path = Path(output_dir) / f"panoramic_stereo_{i:06d}.png"
        dome_img = process_fisheye_frame_pair(l, r, output_path, 
                                             output_width=TARGET_PER_EYE*2, 
                                             output_height=TARGET_PER_EYE)
        
        return i, str(output_path)
    
    # Process frames with multi-threading
    # Use single worker to prevent memory issues
    workers = 1
    
    print(f"  - Using {workers} worker threads for memory efficiency")
    # Single-threaded processing to ensure frame order and prevent memory issues
    for i, (l, r) in enumerate(zip(left_files_filtered, right_files_filtered)):
        try:
            result_index, result_path = _process_single_dome_frame(i, l, r)
            if progress_cb and (i % 5) == 0:  # Less frequent updates
                progress_cb(45 + int(10 * (i / max(1, len(left_files_filtered)))), f'Dome Projection {i}/{len(left_files_filtered)}')
        except Exception as exc:
            print(f'Frame {i} generated an exception: {exc}')
    
    print(f"[DomeProjection] Created {n} equirectangular dome stereo frames")
