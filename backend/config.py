# config.py - True online-only config and parameters
import os
from pathlib import Path
# CPU-only device
DEVICE = "cpu"
# Workspace
PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
TMP_DIR = os.getenv("TMP_DIR", str(Path(os.path.dirname(os.path.abspath(__file__))) / "tmp"))
MODELS_DIR = os.getenv("MODELS_DIR", str(PROJECT_DIR / "models"))  # Not used in online mode
# Deployment optimizations - Settings for online-only processing
INPUT_DOWNSCALE = True
INPUT_TARGET_HEIGHT = 480  # Lower input resolution to save disk space
INPUT_TARGET_HEIGHT_ULTRAFAST = 360  # Ultra-fast mode with lower quality
TARGET_FPS = 24  # Lower frame rate to save disk space
MAX_WORKERS = 1  # Single worker to manage memory usage
KEY_FRAME_INTERVAL = 10  # Less frequent key frames to save processing
# Visual parameters - Enhanced for cinematic "into the scene" experience
IPD_MM = 65.0  # Standard IPD for stereo effect
MAX_DISPARITY_PX = 60  # Reduced for faster processing
PANINI_MIX = 0.6  # Slightly reduced for faster processing
STEREO_MIX = 0.3  # Slightly increased for better projection
IDENTITY_MIX = 0.1  # Keep some identity projection for smooth blend
FOVEATION_START_R = 0.50  # Delay foveation start for faster processing
FOVEATION_MAX_SIGMA = 2  # Reduced blur strength for faster processing
MAX_FOV = 180  # Standard 180 degree FOV
# Resolution targets - Lower resolution to save disk space
PREVIEW_PER_EYE = 1080  # 2K preview resolution
FINAL_PER_EYE = 2160    # 4K resolution (4320x2160 total for 4K VR180)
TARGET_PER_EYE = FINAL_PER_EYE  # Set to 4K resolution
# Output packing
SBS_WIDTH = TARGET_PER_EYE * 2
SBS_HEIGHT = TARGET_PER_EYE
# Super-res options - Using online services instead of local models
USE_REAL_ESRGAN = False  # Disabled since we're using online services
SR_SCALE = 1  # No super-resolution scaling (handled by online services)
# Encoding - Use HEVC encoder for better quality
ENCODER = "libx265"  # Changed from libx264 to libx265 for better quality
CRF = 23  # Higher CRF for faster encoding
PIX_FMT = "yuv420p"  # 8-bit color depth for better performance
# Misc
FPS = 24
# Ultra-fast processing mode - Enable by default for minimal processing
ULTRAFAST_MODE = True  # Enable for faster processing
# Online API settings
ONLINE_API_TIMEOUT = 30  # Timeout for online API calls in seconds
ONLINE_API_RETRIES = 3   # Number of retries for failed API calls