Stellar VR180 Backend (True Online-Only Mode)
==========================================
This package provides a CPU-only VR180 processing backend that uses online API services instead of local AI models, making it extremely lightweight for Git upload and deployment.

Features:
- Depth estimation (MiDaS) - using online API services
- DIBR stereo generation with occlusion masks
- Outpainting (Online API services)
- Panini + stereographic projection to equidistant fisheye
- Foveated edge blur
- Super-resolution (Online API services)
- Pack into side-by-side (SBS) VR180 MP4, inject VR metadata

Benefits:
- Extremely reduced package size (no large model files at all)
- No model downloads required during installation or runtime
- True online-only processing with API services
- Same high-quality output as the full local version
- Easier deployment and Git uploads
- Automatic model updates from upstream API services

How to use:
1. Create venv and install requirements:
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. Configure API keys (if required):
   Set environment variables for online API services

3. Run server:
   uvicorn main:app --host 0.0.0.0 --port 8000
   

4. POST /process with form field 'file' to process a video.

Notes:
- This is CPU-only and uses online APIs for AI processing, making it lightweight for deployment
- No heavy model downloads required - all AI processing is done through online services
- Ultra-fast mode is enabled by default for minimal processing time
- Significantly reduced dependencies for easier deployment
- Internet connection required for AI processing steps
