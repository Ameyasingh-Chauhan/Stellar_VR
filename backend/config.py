# config.py - True online-only config and parameters
import os
from pathlib import Path
# CPU-only device
DEVICE = "cpu"
# Workspace
PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
TMP_DIR = os.getenv("TMP_DIR", str(PROJECT_DIR / "tmp"))
MODELS_DIR = os.getenv("MODELS_DIR", str(PROJECT_DIR / "models"))  # Not used in online mode
# Deployment optimizations - Settings for online-only processing
INPUT_DOWNSCALE = True
INPUT_TARGET_HEIGHT = 720  # Higher input resolution for better quality
INPUT_TARGET_HEIGHT_ULTRAFAST = 480  # Ultra-fast mode with better quality
TARGET_FPS = 30  # Higher frame rate for smoother playback
MAX_WORKERS = 4  # More workers for faster processing
KEY_FRAME_INTERVAL = 5  # More frequent key frames for better quality
# Visual parameters - Enhanced for cinematic "into the scene" experience
IPD_MM = 65.0  # Standard IPD for stereo effect
MAX_DISPARITY_PX = 60  # Increased for stronger cinematic 3D effect
PANINI_MIX = 0.7  # Increased for more cinematic projection
STEREO_MIX = 0.5  # Increased for more cinematic projection
IDENTITY_MIX = 0.0  # Keep focus on cinematic projections
FOVEATION_START_R = 0.40  # Start foveation earlier for more cinematic effect
FOVEATION_MAX_SIGMA = 4  # Increased blur strength for stronger cinematic foveation effect
MAX_FOV = 180  # Standard 180 degree FOV
# Resolution targets - Higher resolution for cinematic quality
PREVIEW_PER_EYE = 1024  # Increased preview resolution
FINAL_PER_EYE = 4096    # 4K per-eye resolution for cinematic quality (can go up to 8192 for 8K)
TARGET_PER_EYE = FINAL_PER_EYE  # Set to 4K per-eye (4096)
# Output packing
SBS_WIDTH = TARGET_PER_EYE * 2
SBS_HEIGHT = TARGET_PER_EYE
# Super-res options - Using online services instead of local models
USE_REAL_ESRGAN = False  # Disabled since we're using online services
SR_SCALE = 1  # No super-resolution scaling (handled by online services)
# Encoding - Use faster encoder
ENCODER = "libx264"  # Faster than libx265
CRF = 23  # Slightly higher compression for smaller files
PIX_FMT = "yuv420p"
# Misc
FPS = 30
# Ultra-fast processing mode - Enable by default for minimal processing
ULTRAFAST_MODE = True  # Enable for fastest possible processing
# Online API settings
ONLINE_API_TIMEOUT = 30  # Timeout for online API calls in seconds
ONLINE_API_RETRIES = 3   # Number of retries for failed API calls