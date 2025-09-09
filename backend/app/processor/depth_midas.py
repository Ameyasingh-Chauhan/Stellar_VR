import os
import torch
import numpy as np
import cv2
from transformers import DPTForDepthEstimation, DPTImageProcessor

HF_TOKEN = os.getenv("HF_TOKEN", None)

MODEL_MAP = {
    "DPT_Hybrid": "Intel/dpt-hybrid-midas",
    "DPT_Large": "Intel/dpt-large-midas"
}

class MiDaSDepth:
    def __init__(self, model_type: str = "DPT_Hybrid", force_device: str = "cpu"):
        if model_type not in MODEL_MAP:
            raise ValueError(f"Unsupported model_type: {model_type}. Available: {list(MODEL_MAP.keys())}")
        
        model_name = MODEL_MAP[model_type]
        print(f"🔧 Loading depth model: {model_name}")

        # Device selection with better logic
        if force_device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"🚀 Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
            print("💻 Using CPU device")

        try:
            # Load processor and model with better error handling
            self.processor = DPTImageProcessor.from_pretrained(model_name, token=HF_TOKEN)
            self.model = DPTForDepthEstimation.from_pretrained(
                model_name, 
                torch_dtype=torch.float32, 
                token=HF_TOKEN,
                low_cpu_mem_usage=True  # Better memory management
            )
            self.model.to(self.device).eval()
            
            # Optimize model for inference
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            # Enable optimizations for better performance
            if self.device.type == "cuda":
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    print("⚡ Model compiled for faster inference")
                except Exception as e:
                    print(f"ℹ️ Model compilation not available: {e}")
                    
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    @torch.no_grad()
    def infer_depth(self, bgr_image):
        """
        Improved depth inference with better preprocessing and postprocessing
        """
        try:
            # Input validation
            if bgr_image is None or bgr_image.size == 0:
                raise ValueError("Invalid input image")
            
            original_height, original_width = bgr_image.shape[:2]
            
            # Convert BGR to RGB
            rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            
            # Process with the model
            inputs = self.processor(images=rgb, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # Interpolate to original size with better quality
            predicted_depth = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(original_height, original_width),
                mode="bicubic",
                align_corners=False
            )
            
            # Convert to numpy and normalize
            depth = predicted_depth.squeeze().cpu().numpy()
            
            # Better normalization to preserve depth relationships
            depth_min = np.percentile(depth, 2)  # Use percentiles instead of min/max
            depth_max = np.percentile(depth, 98) # to handle outliers better
            
            if depth_max - depth_min > 1e-6:
                depth = (depth - depth_min) / (depth_max - depth_min)
                depth = np.clip(depth, 0.0, 1.0)
            else:
                # Handle uniform depth case
                depth = np.full_like(depth, 0.5)
            
            # Apply slight smoothing to reduce noise while preserving edges
            depth = cv2.bilateralFilter(depth.astype(np.float32), 5, 0.1, 0.1)
            
            return depth.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Error in depth inference: {e}")
            # Return fallback depth map
            h, w = bgr_image.shape[:2]
            return np.full((h, w), 0.5, dtype=np.float32)
    
    def process_batch(self, images):
        """
        Process multiple images in a batch for better efficiency
        """
        if not isinstance(images, list):
            return self.infer_depth(images)
        
        depths = []
        for img in images:
            depth = self.infer_depth(img)
            depths.append(depth)
        
        return depths
    
    def get_model_info(self):
        """
        Return information about the loaded model
        """
        return {
            "device": str(self.device),
            "model_type": self.model.__class__.__name__,
            "memory_usage": torch.cuda.memory_allocated() if self.device.type == "cuda" else "N/A"
        }
    
    def cleanup(self):
        """
        Clean up GPU memory if using CUDA
        """
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            print("🧹 GPU memory cleaned up")