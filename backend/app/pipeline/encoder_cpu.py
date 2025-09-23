import subprocess
from pathlib import Path
from .utils import ensure_dir
from config import SBS_WIDTH, SBS_HEIGHT, CRF, PIX_FMT, ENCODER, MAX_WORKERS
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2

def _pack_single_frame_pair(l, r, tmp_frames, i):
    """Pack a single left/right frame pair into side-by-side format"""
    L = cv2.imread(str(l))
    R = cv2.imread(str(r))
    sbs = cv2.hconcat([L, R])
    output_path = tmp_frames / f'sbs_{i:06d}.png'
    cv2.imwrite(str(output_path), sbs)
    return output_path

def pack_and_encode(left_dir, right_dir, output_path, audio_path=None, force_8k=False, progress_cb=None):
    left = sorted(Path(left_dir).glob('*.png'))
    right = sorted(Path(right_dir).glob('*.png'))
    assert len(left) == len(right), 'left/right frame count mismatch'
    tmp_frames = Path(Path(output_path).parent)/'sbs_frames'
    ensure_dir(tmp_frames)
    n = len(left)
    
    if MAX_WORKERS > 1:
        # Use multi-threading for parallel packing
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all packing tasks
            future_to_index = {
                executor.submit(_pack_single_frame_pair, l, r, tmp_frames, i): i 
                for i, (l, r) in enumerate(zip(left, right))
            }
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    completed += 1
                    if progress_cb and (completed % 10) == 0:
                        progress_cb(85 + int(10 * (completed / max(1, n))), f'Packing {completed}/{n}')
                except Exception as exc:
                    print(f'Frame pair {index} generated an exception: {exc}')
    else:
        # Single-threaded packing (fallback)
        for i,(l,r) in enumerate(zip(left,right)):
            L = cv2.imread(str(l))
            R = cv2.imread(str(r))
            sbs = cv2.hconcat([L, R])
            cv2.imwrite(str(tmp_frames / f'sbs_{i:06d}.png'), sbs)
            if progress_cb and i%10==0:
                progress_cb(85 + int(10*(i/n)), f'Packing {i}/{n}')
    
    # encode with ffmpeg and scale to 8K final (16384x8192 for 8K VR180)
    img_pattern = str(tmp_frames / 'sbs_%06d.png')
    final_w = 16384  # 8K width (2 * 8192)
    final_h = 8192   # 8K height
    
    cmd = ['ffmpeg','-y','-framerate','30','-i', img_pattern]
    if audio_path:
        cmd += ['-i', str(audio_path), '-map', '0:v', '-map', '1:a']
    cmd += ['-vf', f'scale={final_w}:{final_h}', '-c:v', ENCODER, '-crf', str(CRF), '-pix_fmt', PIX_FMT]
    if audio_path:
        cmd += ['-c:a', 'aac', '-b:a', '256k']
    cmd += [str(output_path)]
    subprocess.check_call(cmd)

def pack_and_encode_panoramic_stereo(panoramic_dir, output_path, audio_path=None, force_8k=False, progress_cb=None):
    """
    Encode panoramic stereo frames into final VR180 equirectangular format
    Output: Cinematic resolution based on settings (4K or 8K)
    Includes proper spatial encoding for VR180 with cinematic quality
    """
    from config import TARGET_PER_EYE, CRF, PIX_FMT, ENCODER, FPS
    
    panoramic_frames = sorted(Path(panoramic_dir).glob('*.png'))
    n = len(panoramic_frames)
    
    print(f"[PanoramicEncoder] Encoding {n} panoramic stereo frames")
    
    # Use configuration settings instead of hardcoded 8K
    final_w = TARGET_PER_EYE * 2  # Width is 2x eye resolution for stereo
    final_h = TARGET_PER_EYE      # Height is 1x eye resolution
    
    # Determine if we should use 8K or 4K based on settings
    resolution_label = '8K' if final_w >= 15000 else '4K' if final_w >= 7000 else '2K'
    print(f"  - Output resolution: {final_w}x{final_h} ({resolution_label} VR180)")
    print(f"  - Format: Equirectangular stereo side-by-side")
    print(f"  - Projection: 180° horizontal FOV")
    print(f"  - Quality: Cinematic")
    
    if n == 0:
        raise ValueError("No panoramic frames found to encode")
    
    # encode with ffmpeg for panoramic stereo with proper VR180 settings
    img_pattern = str(Path(panoramic_dir) / 'panoramic_stereo_%06d.png')
    in_dir = panoramic_dir
    
    # FFmpeg command with Google spatial metadata for VR180 equirectangular
    cmd = [
        'ffmpeg', '-y', 
        '-framerate', str(FPS),
        '-i', f'{in_dir}/panoramic_stereo_%06d.png',
    ]
    
    # Add audio if available
    if audio_path and Path(audio_path).exists():
        cmd += ['-i', str(audio_path), '-map', '0:v', '-map', '1:a']
    else:
        print("  No audio file found, encoding video only")
    
    # Add video filters for scaling and VR180 metadata
    cmd += [
        '-vf', f'scale={final_w}:{final_h}:flags=lanczos',
        '-c:v', ENCODER,
        '-crf', str(CRF),
        '-pix_fmt', PIX_FMT,
        '-preset', 'slow',  # Slower preset for better quality
        '-tune', 'film',    # Tune for cinematic film quality
        '-profile:v', 'high',
        '-level', '5.2',
    ]
    
    # Add spatial metadata for VR180 equirectangular projection
    cmd += [
        '-metadata:s:v:0', 'stereo_mode=left_right',
        '-metadata:s:v:0', 'projection=equirectangular',
        '-metadata:s:v:0', 'spherical=true',
    ]
    
    # Add audio codec if audio is included
    if audio_path and Path(audio_path).exists():
        cmd += ['-c:a', 'aac', '-b:a', '320k', '-ar', '48000']  # Higher audio bitrate for cinematic quality
    
    # Output file
    cmd += [str(output_path)]
    
    print(f"[PanoramicEncoder] Running FFmpeg command: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    print(f"[PanoramicEncoder] Encoding completed successfully")
