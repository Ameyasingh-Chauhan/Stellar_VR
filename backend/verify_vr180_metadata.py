#!/usr/bin/env python3
"""
VR180 Metadata Verification Script

This script verifies that the output VR180 video has proper spatial metadata
for compatibility with YouTube VR180 and VR headsets.
"""

import subprocess
import json
import sys
from pathlib import Path

def verify_vr180_metadata(video_path):
    """
    Verify that the video has proper VR180 metadata
    """
    print(f"Verifying VR180 metadata for: {video_path}")
    print("=" * 50)
    
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
        '-show_entries', 'format_tags,s stream_tags',
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
    for i, stream in enumerate(streams):
        print(f"  Stream {i}:")
        codec_type = stream.get('codec_type', 'unknown')
        print(f"    Codec Type: {codec_type}")
        tags = stream.get('tags', {})
        for key, value in tags.items():
            print(f"    {key}: {value}")
        print()
    
    # Verify required VR180 metadata
    required_metadata = {
        'stereo_mode': 'left_right',
        'projection': ['equirectangular', 'vr180'],
        'Spherical': 'true',
        'ProjectionType': 'equirectangular',
        'StereoMode': 'left-right'
    }
    
    print("VR180 Metadata Verification:")
    print("-" * 30)
    
    all_checks_passed = True
    
    # Check format-level tags
    for key, expected_values in required_metadata.items():
        if isinstance(expected_values, str):
            expected_values = [expected_values]
            
        found_value = format_tags.get(key, '').lower()
        if found_value in [v.lower() for v in expected_values]:
            print(f"  ✓ {key}: {found_value}")
        else:
            print(f"  ✗ {key}: {found_value} (expected: {expected_values})")
            all_checks_passed = False
    
    # Check stream-level tags
    stream_metadata_found = False
    for stream in streams:
        tags = stream.get('tags', {})
        if 'stereo_mode' in tags or 'projection' in tags:
            stream_metadata_found = True
            for key, expected_values in required_metadata.items():
                if isinstance(expected_values, str):
                    expected_values = [expected_values]
                    
                found_value = tags.get(key, '').lower()
                if found_value in [v.lower() for v in expected_values]:
                    print(f"  ✓ Stream {key}: {found_value}")
                else:
                    print(f"  ✗ Stream {key}: {found_value} (expected: {expected_values})")
                    all_checks_passed = False
    
    if not stream_metadata_found:
        print("  ⚠ No stream-level metadata found")
    
    print()
    print("Additional Checks:")
    print("-" * 20)
    
    # Check resolution
    video_stream = None
    for stream in streams:
        if stream.get('codec_type') == 'video':
            video_stream = stream
            break
    
    if video_stream:
        width = video_stream.get('width', 0)
        height = video_stream.get('height', 0)
        print(f"  Resolution: {width}x{height}")
        
        # Check if it's proper VR180 dimensions (width ≈ 2 * height)
        if width > 0 and height > 0:
            ratio = width / height
            if 1.9 <= ratio <= 2.1:
                print(f"  ✓ Aspect Ratio: {ratio:.2f} (valid VR180)")
            else:
                print(f"  ✗ Aspect Ratio: {ratio:.2f} (should be ~2.0 for VR180)")
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
        print(f"  Codec: {codec}")
        if codec in ['hevc', 'h265']:
            print(f"  ✓ Codec Valid: H.265 (HEVC)")
        elif codec in ['h264']:
            print(f"  ⚠ Codec Valid: H.264 (acceptable)")
        else:
            print(f"  ⚠ Codec Unknown: {codec}")
    
    print()
    if all_checks_passed:
        print("🎉 All VR180 metadata checks PASSED!")
        print("   The video should work correctly in VR headsets and YouTube VR180.")
        return True
    else:
        print("❌ Some VR180 metadata checks FAILED!")
        print("   The video may not work correctly in VR headsets or YouTube VR180.")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_vr180_metadata.py <video_file>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    success = verify_vr180_metadata(video_path)
    sys.exit(0 if success else 1)