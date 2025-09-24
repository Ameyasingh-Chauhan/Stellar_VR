import os, glob, subprocess
from pathlib import Path
import cv2
from config import INPUT_DOWNSCALE, INPUT_TARGET_HEIGHT, TARGET_FPS, ULTRAFAST_MODE, INPUT_TARGET_HEIGHT_ULTRAFAST

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def extract_frames_and_audio(video_path, frames_dir, audio_out):
    ensure_dir(frames_dir)
    
    # Get original video info
    proc = subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=r_frame_rate','-of','default=noprint_wrappers=1:nokey=1', video_path], capture_output=True, text=True)
    orig_fps = 30
    try:
        if proc.stdout:
            num,den = proc.stdout.strip().split('/')
            orig_fps = float(num)/float(den)
    except:
        orig_fps = 30
    
    # Build ffmpeg command for frame extraction with optimizations
    cmd_frames = ['ffmpeg','-y','-i', video_path]
    
    # Combine all video filters into a single -vf parameter to avoid conflicts
    filters = []
    
    # Add frame rate reduction if enabled
    if TARGET_FPS and TARGET_FPS < orig_fps:
        filters.append(f'fps={TARGET_FPS}')
    
    # Add downscaling - ultra-fast mode uses even more aggressive downscaling
    if INPUT_DOWNSCALE and INPUT_TARGET_HEIGHT:
        target_height = INPUT_TARGET_HEIGHT_ULTRAFAST if ULTRAFAST_MODE else INPUT_TARGET_HEIGHT
        filters.append(f"scale=-1:{target_height}")
    
    # Combine all filters
    if filters:
        cmd_frames.extend(['-vf', ','.join(filters)])
    
    cmd_frames.extend(['-vsync','0', f"{frames_dir}/frame_%06d.png"])
    subprocess.check_call(cmd_frames)
    
    cmd_audio = ['ffmpeg','-y','-i', video_path, '-vn','-ac','2','-ar','48000','-f','wav', str(audio_out)]
    subprocess.check_call(cmd_audio)
    
    return min(orig_fps, TARGET_FPS) if TARGET_FPS else orig_fps

def save_img(path, img):
    cv2.imwrite(str(path), img)

def list_frames(frames_dir):
    files = sorted(glob.glob(f"{frames_dir}/*.png"))
    return files

def global_minmax_scan(dir_list, sample_n=5):  # Reduced sample size for faster processing
    import numpy as np, cv2, glob
    mins=[]; maxs=[]
    for d in dir_list:
        files = sorted(glob.glob(f"{d}/*.png"))[:sample_n]
        for f in files:
            im = cv2.imread(f).astype('float32')
            mask = (im.sum(axis=2) > 0)
            if mask.sum() == 0: continue
            px = im[mask]
            mins.append(px.min()); maxs.append(px.max())
    if not mins:
        return 0.0, 255.0
    return float(min(mins)), float(max(maxs))