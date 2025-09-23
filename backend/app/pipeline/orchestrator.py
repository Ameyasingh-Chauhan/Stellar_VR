# orchestrator CPU - full stages with progress
import time
from pathlib import Path
from .utils import ensure_dir, extract_frames_and_audio, list_frames, global_minmax_scan
from .depth_cpu import DepthEstimatorCPU
from .stereo_cpu import StereoCPU
from .outpaint_cpu import OutpaintCPU
from .projection_cpu import process_dirs as projection_process_dirs, create_final_panoramic_stereo_from_fisheye
from .foveate_cpu import foveate_dir
from .sr_cpu import run_sr
from .encoder_cpu import pack_and_encode_panoramic_stereo
from .metadata_cpu import inject_vr180_panoramic_tags
from config import TMP_DIR, MODELS_DIR, MAX_WORKERS, ULTRAFAST_MODE
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np

class VR180PipelineOrchestrator:
    def __init__(self, models_dir=None, tmp_dir=None, device='cpu'):
        self.models_dir = models_dir or MODELS_DIR
        self.tmp_dir = Path(tmp_dir or TMP_DIR)
        ensure_dir(self.tmp_dir)
        self.device = device
        self.depth = DepthEstimatorCPU(device=self.device)
        self.stereo = StereoCPU()
        self.outpaint = OutpaintCPU(device=self.device)
        print(f'[Orchestrator] Ready. device={self.device} models_dir={self.models_dir}')

    def _emit(self, cb, percent, message):
        if cb:
            try:
                cb(stage=message, percent=int(percent))
            except TypeError:
                cb(int(percent), message)

    def _process_depth_frame(self, frame_path, depth_dir):
        """Process a single frame for depth estimation"""
        # In ultra-fast mode, skip some frames for maximum speed
        if ULTRAFAST_MODE:
            frame_name = Path(frame_path).name
            frame_num = int(frame_name.replace('frame_', '').replace('.png', ''))
            # Process every 3rd frame in ultra-fast mode
            if frame_num % 3 != 0:
                # Just copy the previous depth frame or create a blank one
                prev_frame_num = ((frame_num // 3) * 3) - 3
                if prev_frame_num >= 0:
                    prev_depth_path = Path(depth_dir) / f'depth_{prev_frame_num:06d}.png'
                    if prev_depth_path.exists():
                        import shutil
                        outp = Path(depth_dir) / Path(frame_path).name.replace('frame_','depth_')
                        shutil.copy(str(prev_depth_path), str(outp))
                        return outp
                # Create a blank depth frame as fallback
                img = cv2.imread(frame_path)
                blank_depth = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                outp = Path(depth_dir) / Path(frame_path).name.replace('frame_','depth_')
                cv2.imwrite(str(outp), blank_depth)
                return outp
        
        img = cv2.imread(frame_path)
        d = self.depth.estimate_frame(img)
        outp = Path(depth_dir) / Path(frame_path).name.replace('frame_','depth_')
        cv2.imwrite(str(outp), d)
        return outp

    def _process_stereo_frame(self, frame_path, depth_dir, left_dir, right_dir, mask_dir):
        """Process a single frame for stereo generation"""
        # In ultra-fast mode, skip some frames for maximum speed
        if ULTRAFAST_MODE:
            frame_name = Path(frame_path).name
            frame_num = int(frame_name.replace('frame_', '').replace('.png', ''))
            # Process every 3rd frame in ultra-fast mode
            if frame_num % 3 != 0:
                # Just copy the previous stereo frames
                prev_frame_num = ((frame_num // 3) * 3) - 3
                if prev_frame_num >= 0:
                    prev_left_path = Path(left_dir) / f'frame_{prev_frame_num:06d}.png'
                    prev_right_path = Path(right_dir) / f'frame_{prev_frame_num:06d}.png'
                    if prev_left_path.exists() and prev_right_path.exists():
                        import shutil
                        new_frame_name = Path(frame_path).name
                        shutil.copy(str(prev_left_path), str(Path(left_dir)/new_frame_name))
                        shutil.copy(str(prev_right_path), str(Path(right_dir)/new_frame_name))
                        # Create blank mask files
                        blank_mask = np.zeros((cv2.imread(str(prev_left_path)).shape[0], cv2.imread(str(prev_left_path)).shape[1]), dtype=np.uint8)
                        cv2.imwrite(str(Path(mask_dir)/f'mask_l_{new_frame_name}'), blank_mask)
                        cv2.imwrite(str(Path(mask_dir)/f'mask_r_{new_frame_name}'), blank_mask)
                        return new_frame_name
                # Create blank stereo frames as fallback
                img = cv2.imread(frame_path)
                blank_frame = np.zeros_like(img)
                frame_name = Path(frame_path).name
                cv2.imwrite(str(Path(left_dir)/frame_name), blank_frame)
                cv2.imwrite(str(Path(right_dir)/frame_name), blank_frame)
                blank_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                cv2.imwrite(str(Path(mask_dir)/f'mask_l_{frame_name}'), blank_mask)
                cv2.imwrite(str(Path(mask_dir)/f'mask_r_{frame_name}'), blank_mask)
                return frame_name
        
        img = cv2.imread(frame_path)
        depth_img_path = str(Path(depth_dir)/Path(frame_path).name.replace('frame_','depth_'))
        depth_img = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
        L,R,mask_l,mask_r = self.stereo.depth_to_stereo(img, depth_img)
        frame_name = Path(frame_path).name
        cv2.imwrite(str(Path(left_dir)/frame_name), L)
        cv2.imwrite(str(Path(right_dir)/frame_name), R)
        cv2.imwrite(str(Path(mask_dir)/f'mask_l_{frame_name}'), mask_l)
        cv2.imwrite(str(Path(mask_dir)/f'mask_r_{frame_name}'), mask_r)
        return frame_name

    def process_video(self, input_path, output_path, progress_callback=None):
        start_time = time.time()
        jobname = Path(output_path).stem
        workdir = self.tmp_dir / jobname
        ensure_dir(workdir)
        self._emit(progress_callback, 1, 'Extracting frames & audio')
        frames_dir = workdir / 'frames'
        audio_path = workdir / 'audio.wav'
        ensure_dir(frames_dir)
        fps = extract_frames_and_audio(input_path, frames_dir, audio_path)
        frame_files = list_frames(frames_dir)
        total_frames = len(frame_files) or 1
        self._emit(progress_callback, 5, 'Depth estimation (MiDaS)')
        depth_dir = workdir / 'depth'
        ensure_dir(depth_dir)
        
        # In ultra-fast mode, reduce number of workers
        workers = min(MAX_WORKERS, 2) if ULTRAFAST_MODE and MAX_WORKERS > 1 else MAX_WORKERS
        
        # Process depth estimation with more frequent updates
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_frame = {executor.submit(self._process_depth_frame, f, depth_dir): f for f in frame_files}
            completed = 0
            for future in as_completed(future_to_frame):
                frame_path = future_to_frame[future]
                try:
                    result = future.result()
                    completed += 1
                    # More frequent updates
                    if (completed % 5) == 0 or completed == total_frames:
                        pct = 5 + int(10 * (completed / max(1, total_frames)))
                        self._emit(progress_callback, pct, f'Depth: frame {completed}/{total_frames}')
                except Exception as exc:
                    print(f'Frame {frame_path} generated an exception: {exc}')

        self._emit(progress_callback, 18, 'Generating stereo (DIBR) & masks')
        left_dir = workdir / 'left'; right_dir = workdir / 'right'; mask_dir = workdir / 'masks'
        ensure_dir(left_dir); ensure_dir(right_dir); ensure_dir(mask_dir)
        
        # Process stereo generation with more frequent updates
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_frame = {
                executor.submit(self._process_stereo_frame, f, depth_dir, left_dir, right_dir, mask_dir): f 
                for f in frame_files
            }
            completed = 0
            for future in as_completed(future_to_frame):
                frame_path = future_to_frame[future]
                try:
                    result = future.result()
                    completed += 1
                    # More frequent updates
                    if (completed % 5) == 0 or completed == total_frames:
                        pct = 18 + int(7 * (completed / max(1, total_frames)))
                        self._emit(progress_callback, pct, f'DIBR: frame {completed}/{total_frames}')
                except Exception as exc:
                    print(f'Frame {frame_path} generated an exception: {exc}')

        self._emit(progress_callback, 30, 'Outpainting periphery (SD inpaint or fallback)')
        outpaint_dir = workdir / 'outpaint'
        ensure_dir(outpaint_dir)
        self.outpaint.bulk_outpaint(left_dir, outpaint_dir, progress_cb=progress_callback)
        
        self._emit(progress_callback, 45, 'Projection: Panoramic stereo equirectangular (VR180)')
        # Use dual fisheye to equirectangular conversion with FFmpeg v360 filter
        panoramic_dir = workdir / 'panoramic_stereo'
        ensure_dir(panoramic_dir)
        create_final_panoramic_stereo_from_fisheye(str(left_dir), str(right_dir), str(panoramic_dir), progress_cb=progress_callback)
        
        self._emit(progress_callback, 62, 'Applying foveated blur & vignette')
        fov_panoramic = workdir / 'fov_panoramic'
        ensure_dir(fov_panoramic)
        foveate_dir(str(panoramic_dir), str(fov_panoramic), progress_cb=progress_callback)
        
        self._emit(progress_callback, 70, 'Computing global normalization values')
        gm,gM = global_minmax_scan([str(fov_panoramic)], sample_n=10)  # Reduced sample size for faster processing
        self._emit(progress_callback, 72, f'Global min/max: {gm:.2f}/{gM:.2f}')
        
        self._emit(progress_callback, 75, 'Super-resolution (CPU fallback)')
        sr_panoramic = workdir / 'sr_panoramic'
        ensure_dir(sr_panoramic)
        # In ultra-fast mode, skip super-resolution for maximum speed
        if ULTRAFAST_MODE:
            # Just copy the files without super-resolution
            import shutil
            shutil.copytree(fov_panoramic, sr_panoramic, dirs_exist_ok=True)
        else:
            run_sr(str(fov_panoramic), str(sr_panoramic), progress_cb=progress_callback)
        
        self._emit(progress_callback, 90, 'Encoding final panoramic stereo VR180')
        # Use panoramic encoder instead of SBS encoder
        pack_and_encode_panoramic_stereo(str(sr_panoramic), output_path, audio_path=str(audio_path), force_8k=False, progress_cb=progress_callback)
        
        self._emit(progress_callback, 96, 'Injecting VR180 panoramic metadata')
        # Use panoramic metadata injection with proper VR180 tags
        try:
            print(f"[Orchestrator] About to inject VR180 metadata into: {output_path}")
            if not Path(output_path).exists():
                raise FileNotFoundError(f"Output video file not found: {output_path}")
            
            inject_vr180_panoramic_tags(output_path)
            print(f"[Orchestrator] Successfully injected VR180 metadata")
            
            # Verify the metadata was injected by checking the file
            print(f"[Orchestrator] Verifying metadata injection...")
            verify_metadata_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_streams', '-select_streams', 'v:0', str(output_path)
            ]
            
            import subprocess
            result = subprocess.run(verify_metadata_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                import json
                ffprobe_data = json.loads(result.stdout)
                if 'streams' in ffprobe_data and len(ffprobe_data['streams']) > 0:
                    stream = ffprobe_data['streams'][0]
                    if 'tags' in stream:
                        tags = stream['tags']
                        print(f"[Orchestrator] Found metadata tags: {tags}")
                        
                        # Check if the important VR tags are present
                        has_stereo = tags.get('stereo_mode') == 'left-right' or tags.get('STEREO_MODE') == 'left-right'
                        has_spherical = tags.get('spherical') == 'true' or tags.get('SPHERICAL') == 'true'
                        has_projection = tags.get('projection') == 'equirectangular' or tags.get('PROJECTION') == 'equirectangular'
                        
                        if has_stereo and has_spherical and has_projection:
                            print("[Orchestrator] ✅ VR180 metadata verified successfully")
                        else:
                            print("[Orchestrator] ⚠️  VR180 metadata may be incomplete - some tags missing")
                            if not has_stereo:
                                print("  - Missing stereo_mode tag")
                            if not has_spherical:
                                print("  - Missing spherical tag") 
                            if not has_projection:
                                print("  - Missing projection tag")
                    else:
                        print("[Orchestrator] ⚠️  No metadata tags found in video stream")
                else:
                    print("[Orchestrator] ⚠️  No video streams found in output file")
            else:
                print(f"[Orchestrator] ⚠️  Could not verify metadata: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"[Orchestrator] Warning: Metadata verification timed out")
        except FileNotFoundError as e:
            print(f"[Orchestrator] Error: {e}")
        except Exception as e:
            print(f"[Orchestrator] Warning: Failed to inject VR180 metadata: {e}")
            import traceback
            print(f"[Orchestrator] Traceback: {traceback.format_exc()}")
            print("[Orchestrator] Video will still be playable but may not have proper VR metadata")
        
        total = time.time() - start_time
        self._emit(progress_callback, 100, f'Complete - elapsed {total:.1f}s')
        return str(output_path)
