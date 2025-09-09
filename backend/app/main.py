print("🔥 MAIN.PY LOADED - STELLAR VR180 Backend Starting...")

import os
import shutil
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks, Form
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict
import asyncio
import threading

from app.processor.video_processor import FastVR180Processor

# Store active connections and processing status
active_processes: Dict[str, dict] = {}

app = FastAPI(
    title="STELLAR VR180 Backend", 
    description="Advanced VR 180 Video Processing API",
    version="2.0.1"
)

# Enhanced request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    print(f"🌐 {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    # Add comprehensive CORS headers
    response.headers.update({
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS, PUT, DELETE",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Expose-Headers": "Content-Disposition, Content-Length",
        "Access-Control-Max-Age": "3600"
    })
    
    process_time = time.time() - start_time
    print(f"✅ Response: {response.status_code} ({process_time:.2f}s)")
    return response

# Enhanced CORS preflight handler
@app.options("/{path:path}")
async def options_handler(request: Request):
    print(f"📄 CORS Preflight: {request.method} {request.url.path}")
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS, PUT, DELETE",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "3600",
        }
    )

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Length"]
)

# Configuration
TMP_DIR = Path(os.getenv("TMP_DIR", os.path.join(tempfile.gettempdir(), "stellar_vr_backend")))
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Global processor instance
processor_instance = None
processor_lock = threading.Lock()

def get_processor():
    """Get or create processor instance (thread-safe)"""
    global processor_instance
    with processor_lock:
        if processor_instance is None:
            model_choice = os.getenv("MODEL_CHOICE", "DPT_Hybrid")
            force_device = os.getenv("FORCE_DEVICE", "cpu")
            print(f"🚀 Initializing processor: {model_choice} on {force_device}")
            processor_instance = FastVR180Processor(
                model_type=model_choice,
                force_device=force_device
            )
        return processor_instance

def stream_video_file(file_path: Path, chunk_size: int = 8192 * 128):
    """Generator to stream video file in chunks"""
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                yield chunk
    except Exception as e:
        print(f"Error streaming video file: {e}")
        raise

@app.post("/process")
async def process_video(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(..., description="Video file to convert to VR 180"),
    stereo_layout: str = Form(default="left_right", description="Stereo layout: left_right or top_bottom"),
    quality: str = Form(default="medium", description="Output quality: low, medium, high"),
    disparity_intensity: Optional[float] = Form(default=None, description="Stereo disparity strength (1.0-20.0)"),
    target_resolution: Optional[int] = Form(default=None, description="Target resolution height")
):
    """
    Convert a regular video to VR 180 format with proper stereo depth
    Returns a streaming response of the processed video
    """
    
    print("🎬 VR180 PROCESSING REQUEST:")
    print(f"   📁 File: {file.filename} ({file.content_type})")
    print(f"   🎭 Layout: {stereo_layout}")
    print(f"   💎 Quality: {quality}")
    print(f"   📏 Disparity: {disparity_intensity or 14.0}px")
    print(f"   📺 Resolution: {target_resolution or 1440}p")
    
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    supported_formats = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in supported_formats:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format: {file_ext}. Supported: {', '.join(supported_formats)}"
        )

    # Validate parameters
    if stereo_layout not in ["left_right", "top_bottom"]:
        raise HTTPException(status_code=400, detail="stereo_layout must be 'left_right' or 'top_bottom'")
    
    if quality not in ["low", "medium", "high"]:
        raise HTTPException(status_code=400, detail="quality must be 'low', 'medium', or 'high'")

    # Generate unique file paths
    session_id = uuid.uuid4().hex[:8]
    in_path = TMP_DIR / f"{session_id}_input_{file.filename}"
    out_path = TMP_DIR / f"{session_id}_vr180_output.mp4"

    try:
        # Save uploaded file with progress tracking and larger chunks for big files
        print("💾 Saving uploaded file...")
        file_size = 0
        chunk_size = 8192 * 128  # 1MB chunks for better performance with large files
        
        with open(in_path, "wb") as f:
            while chunk := await file.read(chunk_size):
                f.write(chunk)
                await asyncio.sleep(0)  # Let other tasks run
                file_size += len(chunk)
        
        print(f"✅ File saved: {file_size / (1024*1024):.1f} MB")

        # Configure processing parameters
        config = {
            "model_choice": os.getenv("MODEL_CHOICE", "DPT_Hybrid"),
            "target_resolution": target_resolution or int(os.getenv("TARGET_RESOLUTION", "1440")),
            "disparity_intensity": disparity_intensity or float(os.getenv("DISPARITY", "14.0")),
        }
        
        # Quality settings with optimized encoding parameters
        quality_map = {
            "low": {"crf": int(os.getenv("CRF_LOW", "26")), "resolution_scale": 0.75, "preset": "fast"},
            "medium": {"crf": int(os.getenv("CRF_MEDIUM", "21")), "resolution_scale": 1.0, "preset": "medium"},
            "high": {"crf": int(os.getenv("CRF_HIGH", "18")), "resolution_scale": 1.0, "preset": "slow"}
        }
        
        quality_settings = quality_map[quality]
        config["crf"] = quality_settings["crf"]
        config["preset"] = quality_settings["preset"]
        config["target_resolution"] = int(config["target_resolution"] * quality_settings["resolution_scale"])

        # Ensure reasonable resolution limits and even numbers
        config["target_resolution"] = min(config["target_resolution"], 2160)  # Max 4K
        config["target_resolution"] = max(config["target_resolution"], 720)   # Min 720p
        config["target_resolution"] = (config["target_resolution"] // 2) * 2  # Ensure even

        print(f"⚙️ Processing configuration:")
        for key, value in config.items():
            print(f"   {key}: {value}")

        # Get processor instance
        processor = get_processor()

        # Process video with timeout and error handling
        print("🎯 Starting VR 180 conversion...")
        start_time = time.time()
        
        def progress_update(percentage):
            print(f"📊 Progress: {percentage:.1f}%")
            # Store progress for potential WebSocket updates
            active_processes[session_id] = {
                "progress": percentage,
                "status": "processing",
                "start_time": start_time
            }

        try:
            # Call processor with enhanced error handling
            result_path = processor.process_video_fast(
                input_path=str(in_path),
                output_path=str(out_path),
                stereo_layout=stereo_layout,
                disparity_px=config["disparity_intensity"],
                target_resolution=config["target_resolution"],
                crf=config["crf"],
                preset=config["preset"],
                enable_audio=True,
                enable_metadata=True,
                progress_callback=progress_update
            )
        except Exception as processing_error:
            print(f"❌ Processing failed: {processing_error}")
            print(f"Traceback: {traceback.format_exc()}")
            # Cleanup files on error
            background_tasks.add_task(cleanup_files, [in_path, out_path], session_id)
            raise HTTPException(
                status_code=500, 
                detail=f"Video processing failed: {str(processing_error)}"
            )

        processing_time = time.time() - start_time
        print(f"🎉 Processing completed in {processing_time:.1f} seconds!")

        # Validate output
        if not out_path.exists():
            background_tasks.add_task(cleanup_files, [in_path, out_path], session_id)
            raise HTTPException(status_code=500, detail="Output file was not created")

        output_size = out_path.stat().st_size
        if output_size == 0:
            background_tasks.add_task(cleanup_files, [in_path, out_path], session_id)
            raise HTTPException(status_code=500, detail="Output file is empty")

        print(f"📊 Output statistics:")
        print(f"   Size: {output_size / (1024*1024):.1f} MB")
        print(f"   Compression ratio: {(file_size / output_size):.1f}x")

        # Update process status
        active_processes[session_id] = {
            "progress": 100,
            "status": "completed",
            "processing_time": processing_time,
            "output_size": output_size
        }

        # Schedule cleanup after streaming (delayed to allow download)
        background_tasks.add_task(delayed_cleanup, [in_path, out_path], session_id, delay=3600)  # 1 hour delay

        # Prepare response filename
        base_name = Path(file.filename).stem
        response_filename = f"{base_name}_VR180_{stereo_layout}_{quality}.mp4"

        # Prepare headers for streaming response
        headers = {
            "Content-Type": "video/mp4",
            "Content-Disposition": f'inline; filename="{response_filename}"',
            "Content-Length": str(output_size),
            "Accept-Ranges": "bytes",
            "X-Processing-Time": f"{processing_time:.2f}",
            "X-VR180-Layout": stereo_layout,
            "X-Output-Resolution": f"{config['target_resolution']}p",
            "X-Quality": quality,
            "X-Disparity": f"{config['disparity_intensity']}px",
            "Cache-Control": "public, max-age=3600",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Expose-Headers": "Content-Range, Accept-Ranges, Content-Length, Content-Type, Content-Disposition"
        }

        # Return streaming response
        print(f"📤 Starting video stream for {response_filename}")
        return StreamingResponse(
            stream_video_file(out_path),
            media_type="video/mp4",
            headers=headers
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Cleanup on unexpected errors
        background_tasks.add_task(cleanup_files, [in_path, out_path], session_id)
        error_msg = str(e)
        print(f"❌ Processing error: {error_msg}")
        print(f"Full traceback: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500, 
            content={
                "error": "Video processing failed", 
                "detail": error_msg,
                "session_id": session_id
            }
        )

def cleanup_files(file_paths, session_id=None):
    """Enhanced cleanup with session tracking"""
    for file_path in file_paths:
        try:
            if file_path and Path(file_path).exists():
                os.unlink(file_path)
                print(f"🗑️ Cleaned up: {Path(file_path).name}")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
    
    # Remove from active processes
    if session_id and session_id in active_processes:
        del active_processes[session_id]

def delayed_cleanup(file_paths, session_id=None, delay=3600):
    """Cleanup files after a delay to allow downloads"""
    import time
    time.sleep(delay)
    cleanup_files(file_paths, session_id)
    print(f"🗑️ Delayed cleanup completed for session {session_id}")

@app.get("/health")
async def health_check():
    """Enhanced health check with system information"""
    try:
        processor = get_processor()
        model_info = {}
        if hasattr(processor.depth, 'get_model_info'):
            model_info = processor.depth.get_model_info()
        
        # Check available disk space
        import shutil
        disk_usage = shutil.disk_usage(TMP_DIR)
        disk_free_gb = disk_usage.free / (1024**3)
        
        return {
            "status": "healthy", 
            "service": "STELLAR VR180 Backend", 
            "version": "2.0.1", 
            "timestamp": time.time(),
            "system": {
                "temp_dir": str(TMP_DIR), 
                "temp_dir_exists": TMP_DIR.exists(),
                "disk_free_gb": round(disk_free_gb, 2),
                "model_info": model_info,
                "active_processes": len(active_processes)
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=503, 
            content={
                "status": "unhealthy", 
                "error": str(e), 
                "timestamp": time.time()
            }
        )

@app.get("/")
async def root():
    return {
        "service": "STELLAR VR180 Backend API", 
        "version": "2.0.1",
        "endpoints": {
            "/process": "POST - Convert video to VR180",
            "/health": "GET - Health check",
            "/info": "GET - System information",
            "/status/{session_id}": "GET - Processing status"
        }
    }

@app.get("/info")
async def system_info():
    """Enhanced system information"""
    try:
        import torch
        import cv2
        import platform
        import psutil
        
        # Get system resources
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        info = {
            "system": {
                "platform": platform.system(),
                "python_version": f"{platform.python_version()}",
                "opencv_version": cv2.__version__,
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2)
            }
        }
        
        # Add GPU info if available
        if torch.cuda.is_available():
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    "name": gpu_props.name,
                    "memory_gb": round(gpu_props.total_memory / (1024**3), 2),
                    "compute_capability": f"{gpu_props.major}.{gpu_props.minor}"
                })
            info["system"]["gpu_info"] = gpu_info
        
        return info
    except Exception as e:
        return {"error": str(e)}

@app.get("/status/{session_id}")
async def get_status(session_id: str):
    """Get processing status for a session"""
    if session_id in active_processes:
        return active_processes[session_id]
    else:
        return {"status": "not_found", "message": "Session not found or completed"}

@app.on_event("startup")
async def startup_event():
    print("🚀 STELLAR VR180 Backend starting up...")
    print(f"📁 Temp directory: {TMP_DIR}")
    try:
        # Pre-initialize processor to catch any issues early
        get_processor()
        print("✅ Processor initialized successfully")
    except Exception as e:
        print(f"⚠️ Processor init warning: {e}")
        print("System will attempt to initialize processor on first request")

@app.on_event("shutdown")
async def shutdown_event():
    print("🛑 STELLAR VR180 Backend shutting down...")
    global processor_instance
    if processor_instance and hasattr(processor_instance.depth, 'cleanup'):
        try:
            processor_instance.depth.cleanup()
        except Exception as e:
            print(f"Warning during cleanup: {e}")
    
    # Cleanup any remaining temp files
    try:
        temp_files = list(TMP_DIR.glob("*"))
        for temp_file in temp_files:
            try:
                if temp_file.is_file():
                    temp_file.unlink()
            except Exception:
                pass
    except Exception:
        pass
    
    print("👋 Shutdown complete")