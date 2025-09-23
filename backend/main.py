# main.py - CPU-only FastAPI entrypoint
import os, tempfile, uuid, traceback, time
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Apply torchvision compatibility patch before importing other modules
try:
    import sys
    # Try to import the patch module
    import importlib.util
    patch_spec = importlib.util.spec_from_file_location("fix_patch", "fix_torchvision_compatibility.py")
    if patch_spec and patch_spec.loader:
        fix_patch = importlib.util.module_from_spec(patch_spec)
        patch_spec.loader.exec_module(fix_patch)
        if hasattr(fix_patch, 'patch_torchvision_compatibility'):
            fix_patch.patch_torchvision_compatibility()
except Exception as e:
    print(f"Could not apply torchvision patch: {e}")

# Now import the rest of the modules
from app.pipeline.orchestrator import VR180PipelineOrchestrator
from config import TMP_DIR, MODELS_DIR

app = FastAPI(title='STELLAR VR180 Backend (CPU)', version='1.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

# Convert to Path objects
TMP_DIR_PATH = Path(TMP_DIR)
MODELS_DIR_PATH = Path(MODELS_DIR)

# Create directories if they don't exist
TMP_DIR_PATH.mkdir(parents=True, exist_ok=True)
MODELS_DIR_PATH.mkdir(parents=True, exist_ok=True)

# Increase timeout settings - allow up to 1 hour for processing
app.state.processing_tasks = {}

orchestrator = VR180PipelineOrchestrator(models_dir=str(MODELS_DIR_PATH), tmp_dir=str(TMP_DIR_PATH), device='cpu')

def stream_file(path: Path, chunk_size: int = 8192*64):
    with open(path, 'rb') as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            yield data

@app.post('/process')
async def process_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, 'No file uploaded')
    session_id = uuid.uuid4().hex[:8]
    in_path = TMP_DIR_PATH / f"{session_id}_input_{file.filename}"
    out_path = TMP_DIR_PATH / f"{session_id}_vr180_output.mp4"
    with open(in_path, 'wb') as f:
        while chunk := await file.read(1024*1024):
            if not chunk:
                break
            f.write(chunk)
    print(f"Processing {file.filename} session={session_id}")
    start = time.time()
    try:
        def progress_cb(stage=None, percent=None):
            print(f"[session:{session_id}] {percent}% - {stage}")
        orchestrator.process_video(str(in_path), str(out_path), progress_callback=progress_cb)
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(500, str(e))
    elapsed = time.time() - start
    size_mb = out_path.stat().st_size / (1024*1024)
    print(f"Done in {elapsed:.1f}s size={size_mb:.1f}MB")
    return StreamingResponse(stream_file(out_path), media_type='video/mp4', headers={'Content-Disposition': f'inline; filename="{Path(file.filename).stem}_VR180.mp4"'})

# Add endpoint for checking processing status
@app.get('/status/{session_id}')
async def get_status(session_id: str):
    # In a real implementation, you would check the actual processing status
    return JSONResponse({"status": "processing", "message": "Video processing in progress. This may take 10-30 minutes. Please wait..."})

# Increase the timeout for the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=3600,  # 1 hour keep-alive timeout
        timeout_graceful_shutdown=3600  # 1 hour graceful shutdown timeout
    )
