import subprocess, os
from pathlib import Path
import json

def inject_vr180_tags(mp4_path):
    """Inject VR180 metadata for dual-fisheye format (legacy)"""
    tmp = mp4_path + '.meta.mp4'
    cmd = ['ffmpeg','-y','-i', mp4_path, '-c', 'copy', '-metadata:s:v:0', 'stereo_mode=left_right', '-metadata:s:v:0', 'projection=vr180', tmp]
    subprocess.run(cmd, check=True)
    os.replace(tmp, mp4_path)

import subprocess
from pathlib import Path
import json

def inject_vr180_panoramic_tags(video_path):
    """
    Inject proper VR180 panoramic metadata tags for Google Cardboard and other VR players
    Uses FFmpeg to add spatial metadata for equirectangular stereo content
    Includes fallback to Google Spatial Media metadata for maximum compatibility
    """
    print(f"[VR180Metadata] Injecting VR180 panoramic metadata into {video_path}")
    
    # Create a temporary output file
    video_path_obj = Path(video_path)
    temp_path = video_path_obj.parent / f"{video_path_obj.stem}_temp{video_path_obj.suffix}"
    
    # Try FFmpeg standard metadata first (modern approach)
    print("[VR180Metadata] Trying FFmpeg standard metadata approach...")
    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-c', 'copy',  # Copy streams without re-encoding
        '-metadata:s:v:0', 'stereo_mode=left-right',
        '-metadata:s:v:0', 'spherical=true',
        '-metadata:s:v:0', 'projection=equirectangular',
        '-metadata:s:v:0', 'InitialViewFOVDegrees=180',
        str(temp_path)
    ]
    
    try:
        print(f"[VR180Metadata] Running FFmpeg command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
        print(f"[VR180Metadata] FFmpeg standard metadata succeeded")
        print(f"[VR180Metadata] Stdout: {result.stdout[:200]}..." if len(result.stdout) > 200 else f"[VR180Metadata] Stdout: {result.stdout}")
        
        # Replace original file with metadata-injected version
        video_path_obj.unlink()
        temp_path.rename(video_path)
        print("[VR180Metadata] VR180 panoramic metadata injected successfully")
        return
        
    except subprocess.TimeoutExpired as e:
        print(f"[VR180Metadata] FFmpeg standard metadata timed out: {e}")
        # Clean up temp file if it exists
        if temp_path.exists():
            temp_path.unlink()
    except subprocess.CalledProcessError as e:
        print(f"[VR180Metadata] FFmpeg standard metadata failed: {e}")
        print(f"[VR180Metadata] Stderr: {e.stderr[:500]}..." if len(e.stderr) > 500 else f"[VR180Metadata] Stderr: {e.stderr}")
        # Clean up temp file if it exists
        if temp_path.exists():
            temp_path.unlink()
        
        # Fallback to Google Spatial Media metadata (legacy but widely supported)
        print("[VR180Metadata] Falling back to Google Spatial Media metadata")
        temp_path2 = video_path_obj.parent / f"{video_path_obj.stem}_temp2{video_path_obj.suffix}"
        
        # Google Spatial Media metadata for maximum compatibility with all requested tags
        google_spatial_metadata = (
            "Version=1.0,Equirectangular,Stitched=true,"
            "ProjectionType=equirectangular,StereoMode=left-right,"
            "SourceCount=2,InitialViewHeadingDegrees=0,"
            "InitialViewPitchDegrees=0,InitialViewRollDegrees=0,"
            "InitialViewFOVDegrees=180,Timestamp=0"
        )
        
        cmd_fallback = [
            'ffmpeg', '-y',
            '-i', str(video_path),
            '-c', 'copy',  # Copy streams without re-encoding
            '-metadata:s:v:0', f'spherical-video={google_spatial_metadata}',
            '-metadata:s:v:0', 'stereo_mode=left-right',
            '-metadata:s:v:0', 'projection_type=equirectangular',
            '-metadata:s:v:0', 'spherical=true',
            '-metadata:s:v:0', 'InitialViewFOVDegrees=180',
            str(temp_path2)
        ]
        
        try:
            print(f"[VR180Metadata] Running fallback FFmpeg command: {' '.join(cmd_fallback)}")
            result2 = subprocess.run(cmd_fallback, check=True, capture_output=True, text=True, timeout=120)
            print(f"[VR180Metadata] Google Spatial Media metadata succeeded")
            print(f"[VR180Metadata] Stdout: {result2.stdout[:200]}..." if len(result2.stdout) > 200 else f"[VR180Metadata] Stdout: {result2.stdout}")
            
            # Replace original file with metadata-injected version
            video_path_obj.unlink()
            temp_path2.rename(video_path)
            print("[VR180Metadata] Google Spatial Media metadata injected successfully")
            
        except subprocess.TimeoutExpired as e2:
            print(f"[VR180Metadata] Google Spatial Media metadata timed out: {e2}")
            # Clean up temp file if it exists
            if temp_path2.exists():
                temp_path2.unlink()
            raise Exception("Both metadata injection methods timed out")
        except subprocess.CalledProcessError as e2:
            print(f"[VR180Metadata] Google Spatial Media metadata also failed: {e2}")
            print(f"[VR180Metadata] Stderr: {e2.stderr[:500]}..." if len(e2.stderr) > 500 else f"[VR180Metadata] Stderr: {e2.stderr}")
            # Clean up temp file if it exists
            if temp_path2.exists():
                temp_path2.unlink()
            raise Exception("Both metadata injection methods failed")
        except Exception as e2:
            print(f"[VR180Metadata] Unexpected error during fallback metadata injection: {e2}")
            # Clean up temp file if it exists
            if temp_path2.exists():
                temp_path2.unlink()
            raise
    except Exception as e:
        print(f"[VR180Metadata] Unexpected error during metadata injection: {e}")
        # Clean up temp file if it exists
        if temp_path.exists():
            temp_path.unlink()
        raise

def inject_vr180_sbs_tags(video_path):
    """
    Inject VR180 side-by-side metadata for compatibility
    """
    print(f"[VR180SBSMetadata] Injecting VR180 SBS metadata into {video_path}")
    
    # Create a temporary output file
    video_path_obj = Path(video_path)
    temp_path = video_path_obj.parent / f"{video_path_obj.stem}_temp{video_path_obj.suffix}"
    
    # FFmpeg command to add VR180 SBS metadata
    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-c', 'copy',  # Copy streams without re-encoding
        '-metadata', 'stereo_mode=left_right',
        '-metadata', 'spherical=false',
        str(temp_path)
    ]
    
    try:
        print(f"[VR180SBSMetadata] Running FFmpeg command: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        
        # Replace original file with metadata-injected version
        video_path_obj.unlink()
        temp_path.rename(video_path)
        print("[VR180SBSMetadata] VR180 SBS metadata injected successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"[VR180SBSMetadata] Failed to inject VR180 SBS metadata: {e}")
        # Clean up temp file if it exists
        if temp_path.exists():
            temp_path.unlink()
        raise
    except Exception as e:
        print(f"[VR180SBSMetadata] Unexpected error during metadata injection: {e}")
        # Clean up temp file if it exists
        if temp_path.exists():
            temp_path.unlink()
        raise
