import cv2, numpy as np
from pathlib import Path
from .utils import ensure_dir
from config import FOVEATION_START_R, FOVEATION_MAX_SIGMA, MAX_WORKERS, ULTRAFAST_MODE
from concurrent.futures import ThreadPoolExecutor, as_completed

def _process_single_foveation_frame(f, out_dir, start_r, max_sigma):
    """Process a single frame for cinematic foveation with memory-efficient approach"""
    img = cv2.imread(str(f))
    h,w = img.shape[:2]
    cx,cy = w//2,h//2
    R = min(cx,cy)
    
    # Process in smaller chunks for better performance
    chunk_size = min(128, h)  # Larger chunks for better quality
    out = np.zeros_like(img)
    
    for start_row in range(0, h, chunk_size):
        end_row = min(start_row + chunk_size, h)
        chunk_height = end_row - start_row
        
        # Get image chunk
        img_chunk = img[start_row:end_row, :, :].astype('float32')
        
        # Create coordinate grids for this chunk only
        Y_chunk = np.arange(start_row, end_row)
        X_chunk = np.arange(w)
        X_grid, Y_grid = np.meshgrid(X_chunk, Y_chunk, indexing='xy')
        
        # Calculate distance for this chunk
        r_chunk = np.sqrt((X_grid - cx)**2 + (Y_grid - cy)**2)
        
        # Calculate alpha for this chunk
        alpha_chunk = np.clip((r_chunk/R - start_r)/(1.0 - start_r + 1e-8), 0.0, 1.0)
        
        # Process blur in subregions to manage memory and avoid type issues
        out_chunk = img_chunk.copy()
        
        subchunk_size = min(64, chunk_height, w)
        for sub_y in range(0, chunk_height, subchunk_size):
            for sub_x in range(0, w, subchunk_size):
                end_y = min(sub_y + subchunk_size, chunk_height)
                end_x = min(sub_x + subchunk_size, w)
                
                # Get subregions
                img_sub_chunk = img_chunk[sub_y:end_y, sub_x:end_x]
                alpha_sub_chunk = alpha_chunk[sub_y:end_y, sub_x:end_x]
                
                # Calculate average blur strength for this subregion to reduce memory usage
                avg_sigma = float(np.mean(max_sigma * (0.3 + 0.7 * alpha_sub_chunk)))  # Ensure scalar value
                
                # Apply blur with fixed sigma value to avoid OpenCV type error
                if avg_sigma > 0.1:  # Only blur if needed
                    # Ensure kernel size is valid (must be odd and positive)
                    kernel_size = max(1, int(2 * round(avg_sigma * 2) + 1))
                    if kernel_size % 2 == 0:
                        kernel_size += 1  # Make it odd
                    kernel_size = min(kernel_size, 99)  # Limit kernel size
                    
                    blurred_sub_chunk = cv2.GaussianBlur(img_sub_chunk, (kernel_size, kernel_size), avg_sigma)
                    
                    # Apply foveation effect to this subregion
                    alpha_sub_expanded = np.expand_dims(alpha_sub_chunk, axis=2)
                    out_sub_chunk = (1 - alpha_sub_expanded) * img_sub_chunk + alpha_sub_expanded * blurred_sub_chunk
                    out_sub_chunk = np.clip(out_sub_chunk, 0, 255).astype('float32')
                else:
                    out_sub_chunk = img_sub_chunk
                
                # Copy processed subregion to output chunk
                out_chunk[sub_y:end_y, sub_x:end_x] = out_sub_chunk
        
        # Copy processed chunk to output
        out[start_row:end_row, :, :] = out_chunk.astype('uint8')
    
    cv2.imwrite(str(Path(out_dir)/f.name), out)
    return f.name

def foveate_dir(in_dir, out_dir, start_r=FOVEATION_START_R, max_sigma=FOVEATION_MAX_SIGMA, progress_cb=None):
    # In ultra-fast mode, skip foveation entirely for maximum speed
    if ULTRAFAST_MODE:
        ensure_dir(out_dir)
        files = sorted(Path(in_dir).glob('*.png'))
        n = len(files)
        
        # Just copy files without processing
        import shutil
        for i, f in enumerate(files):
            shutil.copy(str(f), str(Path(out_dir)/f.name))
            if progress_cb and i%10==0:
                progress_cb(60 + int(5*(i/n)), f'Copying {i}/{n}')
        return
    
    ensure_dir(out_dir)
    files = sorted(Path(in_dir).glob('*.png'))
    n = len(files)
    
    # Reduce number of workers for memory efficiency
    workers = min(MAX_WORKERS, 2) if MAX_WORKERS > 1 else 1  # Use single worker for memory efficiency
    
    if workers > 1 and len(files) > 5:
        # Use multi-threading for parallel processing
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all foveation tasks
            future_to_file = {
                executor.submit(_process_single_foveation_frame, f, out_dir, start_r, max_sigma): f 
                for f in files
            }
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    completed += 1
                    if progress_cb and (completed % 10) == 0:
                        progress_cb(60 + int(5 * (completed / max(1, n))), f'Foveate {completed}/{n}')
                except Exception as exc:
                    print(f'Frame {file_path} generated an exception: {exc}')
    else:
        # Single-threaded processing (fallback)
        for i, f in enumerate(files):
            img = cv2.imread(str(f))
            h,w = img.shape[:2]
            cx,cy = w//2,h//2
            R = min(cx,cy)
            
            # Process in chunks to avoid memory issues
            chunk_size = min(128, h)  # Smaller chunks for better memory management
            out = np.zeros_like(img)
            
            for start_row in range(0, h, chunk_size):
                end_row = min(start_row + chunk_size, h)
                chunk_height = end_row - start_row
                
                # Get image chunk
                img_chunk = img[start_row:end_row, :, :].astype('float32')
                
                # Create coordinate grids for this chunk only
                Y_chunk = np.arange(start_row, end_row)
                X_chunk = np.arange(w)
                X_grid, Y_grid = np.meshgrid(X_chunk, Y_chunk, indexing='xy')
                
                # Calculate distance for this chunk
                r_chunk = np.sqrt((X_grid - cx)**2 + (Y_grid - cy)**2)
                
                # Calculate alpha for this chunk
                alpha_chunk = np.clip((r_chunk/R - start_r)/(1.0 - start_r + 1e-8), 0.0, 1.0)
                
                # Apply foveation effect to this chunk with memory-efficient approach
                out_chunk = img_chunk.copy()
                
                # Process blur in subregions to manage memory and avoid type issues
                subchunk_size = min(64, chunk_height, w)
                for sub_y in range(0, chunk_height, subchunk_size):
                    for sub_x in range(0, w, subchunk_size):
                        end_y = min(sub_y + subchunk_size, chunk_height)
                        end_x = min(sub_x + subchunk_size, w)
                        
                        # Get subregions
                        img_sub_chunk = img_chunk[sub_y:end_y, sub_x:end_x]
                        alpha_sub_chunk = alpha_chunk[sub_y:end_y, sub_x:end_x]
                        
                        # Calculate average blur strength for this subregion to reduce memory usage
                        avg_sigma = float(np.mean(max_sigma / 2 * alpha_sub_chunk))  # Ensure scalar value
                        
                        # Apply blur with fixed sigma value to avoid OpenCV type error
                        if avg_sigma > 0.1:  # Only blur if needed
                            # Ensure kernel size is valid (must be odd and positive)
                            kernel_size = max(1, int(2 * round(avg_sigma * 2) + 1))
                            if kernel_size % 2 == 0:
                                kernel_size += 1  # Make it odd
                            kernel_size = min(kernel_size, 99)  # Limit kernel size
                            
                            blurred_sub_chunk = cv2.GaussianBlur(img_sub_chunk, (kernel_size, kernel_size), avg_sigma)
                            
                            # Apply foveation effect to this subregion
                            alpha_sub_expanded = np.expand_dims(alpha_sub_chunk, axis=2)
                            out_sub_chunk = (1 - alpha_sub_expanded) * img_sub_chunk + alpha_sub_expanded * blurred_sub_chunk
                            out_sub_chunk = np.clip(out_sub_chunk, 0, 255).astype('float32')
                        else:
                            out_sub_chunk = img_sub_chunk
                        
                        # Copy processed subregion to output chunk
                        out_chunk[sub_y:end_y, sub_x:end_x] = out_sub_chunk
                
                # Copy processed chunk to output
                out[start_row:end_row, :, :] = out_chunk.astype('uint8')
            
            cv2.imwrite(str(Path(out_dir)/f.name), out)
            if progress_cb and i%10==0:
                progress_cb(60 + int(5*(i/n)), f'Foveate {i}/{n}')