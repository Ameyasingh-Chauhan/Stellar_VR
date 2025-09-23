#!/usr/bin/env python3
"""
8K VR180 Verification Script

This script verifies that the output VR180 video has proper 8K resolution and spatial metadata
for compatibility with YouTube VR180 and VR headsets.
"""

import subprocess
import json
import sys
from pathlib import Path

def verify_8k_vr180(video_path):
    """
    Verify that the video has proper 8K VR180 format and metadata
    """
    print(f"Verifying 8K VR180 for: {video_path}")
    print("=" * 60)
    
    # Check if file exists
    if not Path(video_path).exists():
        print(f"Error: File not found: {video_path}")
        return False
    
    # Use ffprobe to extract metadata
    cmd = [
        'ffprobe', 
        '-v', 'quiet', 
        '-print_format', 'json', 
        '-show_streams', 
        '-show_format', 
        '-show_entries', 'format_tags,stream_tags',
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running ffprobe: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"Error parsing ffprobe output: {e}")
        return False
    
    # Check format metadata
    format_tags = metadata.get('format', {}).get('tags', {})
    print("Format Metadata:")
    for key, value in format_tags.items():
        print(f"  {key}: {value}")
    print()
    
    # Check stream metadata
    streams = metadata.get('streams', [])
    print("Stream Metadata:")
    video_stream = None
    audio_stream = None
    
    for i, stream in enumerate(streams):
        print(f"  Stream {i}:")
        codec_type = stream.get('codec_type', 'unknown')
        print(f"    Codec Type: {codec_type}")
        
        if codec_type == 'video':
            video_stream = stream
        elif codec_type == 'audio':
            audio_stream = stream
            
        tags = stream.get('tags', {})
        for key, value in tags.items():
            print(f"    {key}: {value}")
        print()
    
    # Verify 8K VR180 specifications
    print("8K VR180 Verification:")
    print("-" * 25)
    
    all_checks_passed = True
    
    # Check video resolution
    if video_stream:
        width = video_stream.get('width', 0)
        height = video_stream.get('height', 0)
        print(f"  Resolution: {width}x{height}")
        
        # Check if it's proper VR180 dimensions (width ≈ 2 * height for 8K)
        if width > 0 and height > 0:
            # Check for 8K VR180 dimensions
            is_8k = (16000 <= width <= 16500) and (8000 <= height <= 8300)  # 8K VR180 standard
            is_4k = (7500 <= width <= 7800) and (3800 <= height <= 4000)   # 4K VR180 standard
            
            if is_8k:
                print(f"  ✓ 8K VR180 Resolution: {width}x{height} (1:2 aspect ratio)")
            elif is_4k:
                print(f"  ⚠ 4K VR180 Resolution: {width}x{height} (1:2 aspect ratio)")
                print(f"  ⚠ Upgrade to 8K recommended for better quality")
            else:
                print(f"  ✗ Non-standard Resolution: {width}x{height}")
                print(f"  ✗ Expected 8K VR180: ~16384x8192")
                all_checks_passed = False
        else:
            print(f"  ✗ Invalid dimensions: {width}x{height}")
            all_checks_passed = False
    else:
        print("  ✗ No video stream found")
        all_checks_passed = False
    
    # Check codec
    if video_stream:
        codec = video_stream.get('codec_name', '')
        profile = video_stream.get('profile', '')
        print(f"  Codec: {codec} ({profile})")
        
        if codec in ['hevc', 'h265']:
            print(f"  ✓ Codec Valid: H.265 (HEVC)")
        elif codec in ['h264']:
            print(f"  ⚠ Codec Valid: H.265 strongly recommended for 8K VR180")
        else:
            print(f"  ⚠ Unknown codec: {codec}")
    
    # Check frame rate
    if video_stream:
        avg_frame_rate = video_stream.get('avg_frame_rate', '0/0')
        try:
            num, den = avg_frame_rate.split('/')
            fr = float(num) / float(den) if den != '0' else 0
            print(f"  Frame Rate: {fr:.2f} fps")
            
            if 25 <= fr <= 30:
                print(f"  ✓ Standard Frame Rate: {fr:.2f} fps")
            elif 15 <= fr < 25:
                print(f"  ⚠ Lower Frame Rate: Acceptable for VR180")
            else:
                print(f"  ⚠ Non-standard Frame Rate: May cause playback issues")
        except:
            print(f"  ⚠ Could not determine frame rate: {avg_frame_rate}")
    
    # Check color format
    pix_fmt = video_stream.get('pix_fmt', '') if video_stream else ''
    print(f"  Pixel Format: {pix_fmt}")
    
    if pix_fmt in ['yuv420p', 'yuvj420p']:
        print(f"  ✓ Standard Color Format: {pix_fmt}")
    elif pix_fmt:
        print(f"  ⚠ Non-standard Pixel Format: {pix_fmt}")
    else:
        print(f"  ⚠ Unknown Pixel Format")
    
    # Check spatial metadata
    required_metadata = {
        'stereo_mode': ['left-right', 'left_right'],
        'projection': ['equirectangular', 'vr180'],
        'Spherical': ['true'],
        'ProjectionType': ['equirectangular'],
        'StereoMode': ['left-right', 'left_right']
    }
    
    print("\nSpatial Metadata Verification:")
    print("-" * 32)
    
    # Check format-level tags
    metadata_found = 0
    for key, expected_values in required_metadata.items():
        found_value = format_tags.get(key, '').lower()
        if found_value in [v.lower() for v in expected_values]:
            print(f"  ✓ {key}: {found_value}")
            metadata_found += 1
        else:
            # Check stream-level tags
            metadata_at_stream_level = False
            for stream in streams:
                tags = stream.get('tags', {})
                stream_found_value = tags.get(key, '').lower()
                if stream_found_value in [v.lower() for v in expected_values]:
                    print(f"  ✓ {key}: {stream_found_value} (stream)")
                    metadata_found += 1
                    metadata_at_stream_level = True
                    break
            
            if not metadata_at_stream_level:
                print(f"  ✗ {key}: Missing or invalid")
    
    # Check if we have sufficient metadata
    if metadata_found >= 4:
        print(f"  ✓ Spatial metadata: {metadata_found}/5 tags found")
    elif metadata_found >= 2:
        print(f"  ⚠ Spatial metadata: {metadata_found}/5 tags found (may work)")
        all_checks_passed = False
    else:
        print(f"  ✗ Spatial metadata: Insufficient tags found")
        all_checks_passed = False
    
    print()
    print("8K VR180 File Size Check:")
    print("-" * 25)
    
    # Check file size for sanity
    file_size = Path(video_path).stat().st_size / (1024 * 1024)  # Size in MB
    print(f"  File Size: {file_size:.1f} MB")
    
    # Rough estimate for minimum file size for 8K VR180
    # 8K VR180 at 30fps for 1 minute should be roughly 1000-2000MB
    if file_size > 50:  # At least 50MB for a short clip
        print(f"  ✓ File size reasonable for VR180 content")
    else:
        print(f"  ⚠ File size seems small for VR180 content")
    
    print()
    if all_checks_passed:
        print("🎉 All 8K VR180 checks PASSED!")
        print("   The video should work correctly in VR headsets and YouTube VR180.")
        return True
    else:
        print("❌ Some 8K VR180 checks FAILED!")
        print("   The video may not work correctly in VR headsets or YouTube VR180.")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_8k_vr180.py <video_file>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    success = verify_8k_vr180(video_path)
    sys.exit(0 if success else 1)