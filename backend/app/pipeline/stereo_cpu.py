import numpy as np, cv2

class StereoCPU:
    def __init__(self, max_disp_px=60):  # Increased from 45 to 60 for cinematic stereo effect
        self.max_disp = max_disp_px
        print(f'[StereoCPU] Initialized with max disparity: {self.max_disp}px (enhanced for cinematic VR180)')
    
    def depth_to_stereo(self, frame, depth_map):
        """
        Generate stereo left/right views from depth map with enhanced disparity
        Implements depth-based parallax shift for cinematic "inside the scene" effect
        """
        h,w = frame.shape[:2]
        
        # Resize depth_map to match frame dimensions
        depth_map_resized = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Calculate disparity map with enhanced range for cinematic effect
        # Apply non-linear disparity mapping for more pronounced depth perception
        depth_normalized = depth_map_resized.astype('float32') / 255.0
        # Enhanced non-linear mapping for cinematic depth effect
        disp = (1.0 - np.power(depth_normalized, 0.6)) * self.max_disp
        
        # Create coordinate grids
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply depth-based parallax shift with cinematic effect
        # Far objects (low depth values) have less disparity
        # Near objects (high depth values) have more disparity
        map_l_x = (xx + disp/2.0).astype('float32')  # Left eye shifted right
        map_r_x = (xx - disp/2.0).astype('float32')  # Right eye shifted left
        map_y = yy.astype('float32')
        
        # Generate stereo views with cinematic disparity
        left = cv2.remap(frame, map_l_x, map_y, interpolation=cv2.INTER_LINEAR, 
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        right = cv2.remap(frame, map_r_x, map_y, interpolation=cv2.INTER_LINEAR, 
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # Create occlusion masks for invalid regions
        mask_l = ((left.sum(axis=2) == 0)).astype('uint8')*255
        mask_r = ((right.sum(axis=2) == 0)).astype('uint8')*255
        
        return left, right, mask_l, mask_r
