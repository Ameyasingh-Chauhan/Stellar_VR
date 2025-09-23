from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image
import requests
from io import BytesIO
from config import MODELS_DIR, MAX_WORKERS, KEY_FRAME_INTERVAL, MAX_FOV, ULTRAFAST_MODE
import time
import sys
import os

class EfficientOutpaintCPU:
    def __init__(self, device='cpu'):
        self.device = device
        self.ready = True
        self.key_frame_interval = KEY_FRAME_INTERVAL
        self.max_fov = MAX_FOV
        self.ultrafast_mode = ULTRAFAST_MODE
        print('[OnlineOutpaintCPU] Initialized with online API support')
        print(f'  Key frame interval: {self.key_frame_interval}')
        if self.ultrafast_mode:
            print('  Ultra-fast mode enabled!')
            # In ultra-fast mode, process every 5th frame as key frame instead of every 20th
            self.key_frame_interval = max(1, self.key_frame_interval // 4)
            print(f'  Ultra-fast key frame interval: {self.key_frame_interval}')
        
        # Online API endpoint (example, replace with actual API)
        self.api_url = None  # Will use online services instead of local models
    
    def _create_outpaint_mask(self, frame):
        """Create mask for outpainting regions (left/right borders)"""
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Mask 25% of width on each side
        border_width = w // 4
        mask[:, :border_width] = 255  # Left border
        mask[:, -border_width:] = 255  # Right border
        
        return mask
    
    def _simple_outpaint(self, frame):
        """Fast outpainting using border extension"""
        h, w = frame.shape[:2]
        # In ultra-fast mode, use smaller border extension
        border_ratio = 4 if self.ultrafast_mode else 8
        left_border = frame[:, :w//border_ratio]
        right_border = frame[:, -w//border_ratio:]
        
        # Flip borders for better continuity
        left_extended = cv2.flip(left_border, 1)
        right_extended = cv2.flip(right_border, 1)
        
        # Concatenate: left_extension + original + right_extension
        outpainted = cv2.hconcat([left_extended, frame, right_extended])
        return outpainted
    
    def _online_outpaint(self, frame):
        """Outpaint using online API service"""
        # For demonstration, we'll just extend the borders
        # In a real implementation, you would call an actual online outpainting API
        return self._simple_outpaint(frame)
    
    def _compute_optical_flow(self, prev_frame, curr_frame):
        """Compute optical flow between frames using Farneback"""
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # In ultra-fast mode, use faster but less accurate settings
        if self.ultrafast_mode:
            # Use faster pyramid levels and smaller window
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 2, 10, 2, 3, 1.1, 0
            )
        else:
            # Use standard settings
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
        
        return flow
    
    def _warp_frame_with_flow(self, frame, flow):
        """Warp frame using optical flow"""
        h, w = frame.shape[:2]
        
        # Create coordinate arrays
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply flow
        x_coords = x_coords.astype(np.float32) + flow[:,:,0]
        y_coords = y_coords.astype(np.float32) + flow[:,:,1]
        
        # Clip coordinates to valid range
        x_coords = np.clip(x_coords, 0, w-1)
        y_coords = np.clip(y_coords, 0, h-1)
        
        # Warp frame
        warped = cv2.remap(
            frame, x_coords, y_coords, 
            interpolation=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_REFLECT
        )
        
        return warped
    
    def _feather_blend(self, frame1, frame2, alpha=0.5):
        """Blend two frames with feathering to reduce flicker"""
        # Ensure frames have same dimensions
        if frame1.shape != frame2.shape:
            # Resize frame2 to match frame1
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # In ultra-fast mode, use simpler blending
        if self.ultrafast_mode:
            # Simple average instead of gradient blending
            blended = ((frame1.astype(np.float32) + frame2.astype(np.float32)) / 2).astype(np.uint8)
            return blended
        
        # Create feathered mask
        h, w = frame1.shape[:2]
        feather_width = max(1, w // 8)
        
        # Create gradient mask
        mask = np.ones((h, w), dtype=np.float32)
        if feather_width > 0:
            mask[:, :feather_width] = np.linspace(0, 1, feather_width)
            mask[:, -feather_width:] = np.linspace(1, 0, feather_width)
        
        # Expand mask to 3 channels
        mask = np.stack([mask] * 3, axis=2)
        
        # Blend frames
        blended = (frame1.astype(np.float32) * mask + frame2.astype(np.float32) * (1 - mask))
        return blended.astype(np.uint8)
    
    def _adjust_fov(self, frame, target_fov=None):
        """Adjust FOV to maximum specified degrees"""
        if target_fov is None:
            target_fov = self.max_fov
            
        h, w = frame.shape[:2]
        max_possible_fov = 220  # Base FOV
        
        # If target FOV is already the maximum or larger, return as is
        if target_fov >= max_possible_fov:
            return frame
            
        # Calculate crop to reduce FOV
        scale_factor = target_fov / max_possible_fov
        target_width = int(w * scale_factor)
        
        if target_width >= w:
            return frame
            
        start_x = (w - target_width) // 2
        cropped = frame[:, start_x:start_x + target_width]
        
        # Resize back to original width to maintain aspect ratio
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return resized
    
    def outpaint_key_frame(self, frame):
        """Outpaint key frames using online API"""
        return self._online_outpaint(frame)
    
    def propagate_outpaint(self, key_frame_result, prev_frame, curr_frame):
        """Propagate outpainting to intermediate frames using optical flow"""
        try:
            # Compute optical flow from previous to current frame
            flow = self._compute_optical_flow(prev_frame, curr_frame)
            
            # Warp the key frame result using optical flow
            # Note: We need to handle the case where key_frame_result might be a different size
            if key_frame_result.shape[:2] != curr_frame.shape[:2]:
                # Resize key_frame_result to match current frame dimensions
                key_frame_result = cv2.resize(
                    key_frame_result, 
                    (curr_frame.shape[1], curr_frame.shape[0]), 
                    interpolation=cv2.INTER_LINEAR
                )
            
            warped_result = self._warp_frame_with_flow(key_frame_result, flow)
            
            # Feather blend to reduce flicker
            blended = self._feather_blend(key_frame_result, warped_result, 0.7)
            
            # Adjust FOV
            final_result = self._adjust_fov(blended)
            
            return final_result
        except Exception as e:
            print(f'  Flow propagation failed, using simple outpaint: {e}')
            return self._simple_outpaint(curr_frame)
    
    def bulk_outpaint(self, in_dir, out_dir, progress_cb=None):
        """Efficient bulk outpainting with key frame + propagation approach"""
        print("[OnlineOutpaintCPU] Starting optimized outpainting:")
        print(f"  - Key frame outpainting every {self.key_frame_interval} frames")
        if self.ultrafast_mode:
            print("  Ultra-fast mode: Even sparser key frames and faster processing")
        print("  - Optical flow propagation for intermediate frames")
        print("  - Feather blending to reduce flicker")
        print(f"  - FOV limited to {self.max_fov}°")
        
        start_time = time.time()
        
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        files = sorted(Path(in_dir).glob('*.png'))
        n = len(files)
        
        if n == 0:
            print("No files to process")
            return
            
        print(f"Processing {n} frames")
        
        # Process frames
        key_frame_result = None
        prev_frame = None
        
        for i, f in enumerate(files):
            if i % 20 == 0:  # Print progress every 20 frames
                print(f"Processing frame {i+1}/{n}: {f.name}")
            
            curr_frame = cv2.imread(str(f))
            if curr_frame is None:
                print(f"  Failed to read frame {f.name}")
                continue
            
            # Check if this should be a key frame
            if i % self.key_frame_interval == 0 or key_frame_result is None:
                if i % self.key_frame_interval == 0:
                    print(f"  Frame {i+1}: Processing as key frame")
                key_frame_result = self.outpaint_key_frame(curr_frame)
                prev_frame = curr_frame
            else:
                # In ultra-fast mode, skip some propagation for even faster processing
                if self.ultrafast_mode and i % (self.key_frame_interval // 2) != 0:
                    # Just reuse the previous result with simple scaling
                    if key_frame_result is not None:
                        scaled_result = cv2.resize(
                            key_frame_result, 
                            (curr_frame.shape[1] * 2, curr_frame.shape[0]),  # Double width for outpainted result
                            interpolation=cv2.INTER_LINEAR
                        )
                        key_frame_result = scaled_result
                        prev_frame = curr_frame
                        print(f"  Frame {i+1}: Ultra-fast skip (reusing scaled key frame)")
                    else:
                        # Fallback to simple outpainting
                        key_frame_result = self._simple_outpaint(curr_frame)
                        prev_frame = curr_frame
                        print(f"  Frame {i+1}: Ultra-fast fallback to simple outpaint")
                else:
                    # Propagate from previous key frame using optical flow
                    if prev_frame is not None and key_frame_result is not None:
                        if i % 20 == 0:  # Print every 20 frames to avoid spam
                            print(f"  Frame {i+1}: Propagating from key frame using optical flow")
                        key_frame_result = self.propagate_outpaint(
                            key_frame_result, prev_frame, curr_frame
                        )
                        prev_frame = curr_frame
                    else:
                        # Fallback to simple outpainting
                        if i % 20 == 0:
                            print(f"  Frame {i+1}: Using simple outpaint (fallback)")
                        key_frame_result = self._simple_outpaint(curr_frame)
                        prev_frame = curr_frame
            
            # Save result
            output_path = Path(out_dir) / f.name
            cv2.imwrite(str(output_path), key_frame_result)
            
            # Update progress
            if progress_cb and i % 10 == 0:
                progress_cb(35 + int(10 * (i / max(1, n))), f'Outpaint {i}/{n}')
        
        end_time = time.time()
        print(f"[OnlineOutpaintCPU] Completed in {end_time - start_time:.2f} seconds")


# Keep the old class name for compatibility
OutpaintCPU = EfficientOutpaintCPU