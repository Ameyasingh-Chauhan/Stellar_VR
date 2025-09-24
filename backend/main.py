import os, uuid, tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import shutil
from app.pipeline.orchestrator import VR180PipelineOrchestrator

app = FastAPI(title="VR180 Pipeline API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use a single temp directory for everything to avoid cross-device issues
TEMP_DIR = Path("./tmp").resolve()
TEMP_DIR.mkdir(exist_ok=True)

# Initialize orchestrator with the same temp directory
orchestrator = VR180PipelineOrchestrator(tmp_dir=str(TEMP_DIR))

@app.post("/process")
async def process_video(file: UploadFile):
    # Generate unique session ID
    session_id = str(uuid.uuid4())[:8]
    print(f"[session:{session_id}] Received upload: {file.filename}")
    
    # Create session directories in the same temp directory
    session_dir = TEMP_DIR / f"{session_id}_vr180_output"
    session_dir.mkdir(exist_ok=True)
    
    in_path = session_dir / f"input_{file.filename}"
    out_path = session_dir / f"output_{Path(file.filename).stem}_VR180.mp4"
    
    # Save uploaded file
    with open(in_path, "wb") as f:
        while chunk := await file.read(8192):
            f.write(chunk)
    
    print(f"[session:{session_id}] Saved input to: {in_path}")
    
    # Process video
    def progress_cb(stage, percent):
        print(f"[session:{session_id}] {stage} - {percent}%")
    
    try:
        result_path = orchestrator.process_video(str(in_path), str(out_path), progress_callback=progress_cb)
        print(f"[session:{session_id}] Processing completed: {result_path}")
        # Return the actual video file
        return FileResponse(
            str(result_path),
            media_type="video/mp4",
            filename=f"{Path(file.filename).stem}_VR180.mp4"
        )
    except Exception as e:
        print(f"[session:{session_id}] Processing failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup input file immediately to save space
        try:
            in_path.unlink()
        except:
            pass

# Serve static files (for downloading results)
app.mount("/", StaticFiles(directory=str(TEMP_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)