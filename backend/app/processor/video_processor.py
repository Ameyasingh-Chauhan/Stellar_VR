import cv2
import numpy as np
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from tqdm import tqdm
from typing import Union, Optional, Callable
import time
import threading
import queue

from app.processor.stereoscopic import synthesize_stereo
from app.processor.depth_midas import MiDaSDepth


class FastVR180Processor:
    def __init__(self, model_type="DPT_Hybrid", force_device="cpu"):
        self.depth = MiDaSDepth(model_type=model_type, force_device=force_device)

    def _input_has_audio(self, path: Union[str, Path]) -> bool:
        """Return True if input file contains an audio stream (uses ffprobe)."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-select_streams", "a:0", 
                "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return "audio" in result.stdout.strip()
        except Exception:
            return False

    def _get_video_info(self, path: Union[str, Path]) -> dict:
        """Get detailed video information using ffprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", str(path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                
                video_stream = None
                audio_stream = None
                
                for stream in data.get('streams', []):
                    if stream.get('codec_type') == 'video' and not video_stream:
                        video_stream = stream
                    elif stream.get('codec_type') == 'audio' and not audio_stream:
                        audio_stream = stream
                
                return {
                    'video_stream': video_stream,
                    'audio_stream': audio_stream,
                    'format': data.get('format', {}),
                    'duration': float(data.get('format', {}).get('duration', 0))
                }
        except Exception as e:
            print(f"Warning: Could not get video info: {e}")
        
        return {'video_stream': None, 'audio_stream': None, 'format': {}, 'duration': 0}

    def _log_ffmpeg_output(self, proc, output_queue):
        """Log FFmpeg output in a separate thread to prevent buffer overflow."""
        try:
            for line in iter(proc.stderr.readline, b''):
                line_str = line.decode('utf-8', errors='ignore').strip()
                if line_str:
                    output_queue.put(line_str)
        except Exception:
            pass
        finally:
            output_queue.put(None)  # Signal end

    def process_video_fast(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        disparity_px: float = 14.0,
        progress_callback: Optional[Callable[[int], None]] = None,
        stereo_layout: str = "left_right",
        target_resolution: int = 1920,
        crf: int = 20,
        preset: str = "medium",
        enable_audio: bool = True,
        enable_metadata: bool = True,
    ) -> str:
        """
        Fixed single-pass pipeline for VR180 video processing.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Force left_right layout for VR180
        if stereo_layout != "left_right":
            print("Warning: Only 'left_right' layout supported for VR180. Using left_right.")
            stereo_layout = "left_right"

        # Ensure even resolution
        target_resolution = (target_resolution // 2) * 2

        # Get video information
        video_info = self._get_video_info(input_path)
        video_stream = video_info.get('video_stream')
        audio_stream = video_info.get('audio_stream')
        
        if not video_stream:
            raise RuntimeError("No video stream found in input file")

        # Extract video properties
        source_width = int(video_stream.get('width', 1920))
        source_height = int(video_stream.get('height', 1080))
        source_fps = float(video_stream.get('r_frame_rate', '30/1').split('/')[0]) / float(video_stream.get('r_frame_rate', '30/1').split('/')[1])
        
        # Get frame count from duration and fps
        duration = video_info.get('duration', 0)
        frame_count = int(duration * source_fps) if duration > 0 else 1000

        print(f"Video Info - Resolution: {source_width}x{source_height}, FPS: {source_fps:.2f}, Duration: {duration:.1f}s")
        print(f"Audio present: {audio_stream is not None}")

        # VR180 output dimensions (2:1 aspect ratio)
        output_height = target_resolution
        output_width = target_resolution * 2
        
        # Ensure even dimensions for codec compatibility
        output_width = (output_width // 2) * 2
        output_height = (output_height // 2) * 2

        print(f"Output dimensions: {output_width}x{output_height}")

        # Open video capture
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open input video: {input_path}")

        try:
            # Build FFmpeg command for single-pass encoding
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{output_width}x{output_height}",
                "-r", f"{source_fps:.3f}",
                "-i", "-",  # Video from stdin
            ]

            # Add audio input if present
            if enable_audio and audio_stream:
                ffmpeg_cmd.extend(["-i", str(input_path)])  # Audio from original file
                ffmpeg_cmd.extend(["-map", "0:v:0", "-map", "1:a:0"])  # Map video from stdin, audio from file
            else:
                ffmpeg_cmd.extend(["-map", "0:v:0"])  # Only video

            # Video encoding settings optimized for VR180 web playback
            ffmpeg_cmd.extend([
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", str(crf),
                "-preset", preset,
                "-profile:v", "main",          # More compatible profile for web
                "-level", "4.0",               # Widely supported level
                "-x264-params", "ref=2:bframes=2:keyint=60:min-keyint=30:scenecut=40:rc-lookahead=40",
                "-movflags", "+faststart+frag_keyframe+empty_moov+default_base_moof",
                "-f", "mp4",                   # Force MP4 format
            ])

            # Audio encoding settings
            if enable_audio and audio_stream:
                ffmpeg_cmd.extend([
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-ar", "48000",
                    "-ac", "2",
                ])

            # Add comprehensive VR180 metadata
            if enable_metadata:
                ffmpeg_cmd.extend([
                    "-metadata", "stereo_mode=left_right",
                    "-metadata:s:v:0", "stereo_mode=left_right",
                    "-metadata:s:v:0", "stereo3d_type=sbs2l",
                    "-metadata", "spherical-video=true",
                    "-metadata:s:v:0", "projection_type=equirectangular",
                    "-metadata", "youtube_vr_180=true",
                    "-metadata:s:v:0", "spatial-video=true",
                    "-metadata:s:v:0", "spatial-format=mesh",
                    "-metadata:s:v:0", "spatial-arrangement=side-by-side",
                ])

            ffmpeg_cmd.append(str(output_path))

            print("Starting FFmpeg process...")
            print(f"Command: {' '.join(ffmpeg_cmd[:10])}...")

            # Start FFmpeg process with proper buffer management
            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0  # Unbuffered for real-time processing
            )

            # Create output logging thread
            output_queue = queue.Queue()
            log_thread = threading.Thread(target=self._log_ffmpeg_output, args=(proc, output_queue))
            log_thread.daemon = True
            log_thread.start()

            # Process frames
            frame_idx = 0
            last_progress = 0
            start_time = time.time()

            if progress_callback:
                progress_callback(0)

            print("Processing frames...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached")
                    break

                frame_idx += 1
                
                # Progress update
                if frame_count > 0:
                    current_progress = min(90, int((frame_idx / frame_count) * 90))
                    if current_progress > last_progress:
                        last_progress = current_progress
                        if progress_callback:
                            progress_callback(current_progress)

                # Resize frame to maintain aspect ratio
                h, w = frame.shape[:2]
                scale = min(output_width / (2 * w), output_height / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                # Ensure even dimensions
                new_w = (new_w // 2) * 2
                new_h = (new_h // 2) * 2
                
                frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

                # Generate depth map
                depth_map = self.depth.infer_depth(frame_resized)

                # Create stereo pair
                left_eye, right_eye = synthesize_stereo(
                    frame_resized, depth_map, 
                    disparity_px=disparity_px, 
                    smooth_edges=True
                )

                # Create side-by-side stereo frame
                stereo_frame = np.hstack([left_eye, right_eye])

                # Resize to exact output dimensions
                if stereo_frame.shape[1] != output_width or stereo_frame.shape[0] != output_height:
                    stereo_frame = cv2.resize(stereo_frame, (output_width, output_height), 
                                            interpolation=cv2.INTER_LINEAR)

                # Ensure proper data type and memory layout
                if stereo_frame.dtype != np.uint8:
                    stereo_frame = np.clip(stereo_frame, 0, 255).astype(np.uint8)
                
                if not stereo_frame.flags['C_CONTIGUOUS']:
                    stereo_frame = np.ascontiguousarray(stereo_frame)

                # Write frame to FFmpeg
                try:
                    proc.stdin.write(stereo_frame.tobytes())
                    proc.stdin.flush()
                except BrokenPipeError:
                    print("FFmpeg pipe closed early")
                    break
                except Exception as e:
                    print(f"Error writing to FFmpeg: {e}")
                    break

                # Check if FFmpeg process is still running
                if proc.poll() is not None:
                    print("FFmpeg process terminated early")
                    break

                # Log progress occasionally
                if frame_idx % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_idx / elapsed if elapsed > 0 else 0
                    print(f"Processed {frame_idx} frames, {fps:.1f} fps")

        finally:
            cap.release()
            
            # Close stdin to signal end of input
            if proc.stdin:
                try:
                    proc.stdin.close()
                except:
                    pass

        # Wait for FFmpeg to complete
        print("Waiting for FFmpeg to finish...")
        
        try:
            stdout, stderr = proc.communicate(timeout=60)
            
            # Process any remaining log output
            while True:
                try:
                    line = output_queue.get_nowait()
                    if line is None:
                        break
                    if "error" in line.lower() or "failed" in line.lower():
                        print(f"FFmpeg: {line}")
                except queue.Empty:
                    break
            
        except subprocess.TimeoutExpired:
            print("FFmpeg timeout - terminating process")
            proc.kill()
            stdout, stderr = proc.communicate()

        # Check FFmpeg result and validate output
        if proc.returncode != 0:
            stderr_text = stderr.decode('utf-8', errors='ignore') if stderr else ""
            print(f"FFmpeg failed with return code {proc.returncode}")
            print(f"FFmpeg stderr: {stderr_text[:1000]}")
            raise RuntimeError(f"FFmpeg encoding failed: {stderr_text[:500]}")
            
        # Verify the output video format and validity
        try:
            validation_cmd = ["ffprobe", "-v", "error", str(output_path)]
            validation = subprocess.run(validation_cmd, capture_output=True, text=True)
            if validation.returncode != 0:
                print(f"Warning: Output validation failed: {validation.stderr}")
                print("Attempting to verify basic file properties...")
        except subprocess.SubprocessError as e:
            print(f"Warning: Could not validate output with ffprobe: {e}")
        except Exception as e:
            print(f"Warning: Unexpected error during validation: {e}")

        # Verify output file exists and has content
        if not output_path.exists():
            raise RuntimeError("Output file was not created")

        file_size = output_path.stat().st_size
        if file_size == 0:
            raise RuntimeError("Output file is empty")

        # Final progress
        if progress_callback:
            progress_callback(100)

        processing_time = time.time() - start_time
        print(f"Processing completed in {processing_time:.1f} seconds")
        print(f"Output file: {output_path} ({file_size / (1024*1024):.1f} MB)")
        print(f"Processed {frame_idx} frames at {frame_idx/processing_time:.1f} fps average")

        return str(output_path)