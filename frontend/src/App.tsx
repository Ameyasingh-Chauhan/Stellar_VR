import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Sparkles, X, CheckCircle2, Download, Film, Sun, Moon, Upload, Rocket, Zap } from 'lucide-react';

// Match your backend URL exactly
const API_URL = 'http://127.0.0.1:8000';
const WS_URL = 'ws://127.0.0.1:8000/ws';

interface ProcessVideoParams {
  file: File;
  stereoLayout: 'side-by-side' | 'over-under';
  quality: 'high' | 'medium' | 'low';
}

interface ProcessVideoResponse {
  success: boolean;
  data?: Blob;
  error?: string;
}

/**
 * Sends the video file and options to the backend for processing.
 */
const processVideo = async (params: ProcessVideoParams, onProgress?: (progress: number) => void): Promise<ProcessVideoResponse> => {
  const { file, stereoLayout, quality } = params;

  const formData = new FormData();
  formData.append('file', file);
  formData.append('stereo_layout', stereoLayout === 'side-by-side' ? 'left_right' : 'top_bottom');
  formData.append('quality', quality);

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 14400000); // 4 hour timeout

    const response = await fetch(`${API_URL}/process`, {
      method: 'POST',
      body: formData,
      headers: {
        'Accept': 'video/mp4,*/*',
        'X-File-Size': file.size.toString(),
      },
      signal: controller.signal
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorText = await response.text();
      try {
        const errorData = JSON.parse(errorText);
        throw new Error(errorData.error || `Server error: ${response.status}`);
      } catch {
        throw new Error(`Server error: ${response.status}`);
      }
    }

    const contentType = response.headers.get('content-type');
    if (!contentType?.includes('video/') && !contentType?.includes('application/octet-stream')) {
      throw new Error('Server returned invalid content type');
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body available');
    }

    const chunks: Uint8Array[] = [];
    let receivedLength = 0;
    const contentLength = parseInt(response.headers.get('content-length') || '0', 10);

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      receivedLength += value.length;
      
      if (contentLength && onProgress) {
        const percentComplete = Math.min(95 + ((receivedLength / contentLength) * 5), 99);
        onProgress(percentComplete);
      }
    }

    const blob = new Blob(chunks, { type: 'video/mp4' });
    if (blob.size === 0) {
      throw new Error('Received empty video response');
    }

    return { success: true, data: blob };
  } catch (error: unknown) {
    console.error("❌ Network or fetch error:", error);
    return { 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error'
    };
  }
};

// --- Component Types ---
type Status = 'idle' | 'uploading' | 'processing' | 'success' | 'error';
type StereoLayout = 'side-by-side' | 'over-under';
type Quality = 'high' | 'medium' | 'low';
type Theme = 'dark' | 'light';

interface AppState {
  file: File | null;
  status: Status;
  progress: number;
  stereoLayout: StereoLayout;
  quality: Quality;
  processedVideoUrl: string | null;
  error: string | null;
}

const VR_FACTS = [
  "🎬 VR180 creates an immersive 180-degree field of view",
  "🧠 AI algorithms enhance depth perception in converted videos", 
  "🚀 VR market is expected to reach $87B by 2030",
  "👁️ Human vision spans approximately 180 degrees horizontally",
  "🎮 VR reduces motion sickness through better frame rates",
  "🔬 Neural networks can predict 3D depth from 2D images",
  "🌟 VR180 maintains higher quality than full 360-degree video",
  "⚡ Modern GPUs can process VR content at 90+ FPS",
];

interface WebSocketMessage {
  progress: number;
  status: 'processing' | 'complete' | 'error';
}

const App: React.FC = () => {
  const [theme, setTheme] = useState<Theme>('dark');
  const [state, setState] = useState<AppState>({
    file: null,
    status: 'idle',
    progress: 0,
    stereoLayout: 'side-by-side',
    quality: 'high',
    processedVideoUrl: null,
    error: null,
  });
  const [inputVideoUrl, setInputVideoUrl] = useState<string | null>(null);
  const [currentFactIndex, setCurrentFactIndex] = useState(0);
  const [rotation, setRotation] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // WebSocket connection management
  const wsRef = useRef<WebSocket | null>(null);
  const clientId = useRef(Math.random().toString(36).substring(7));

  // Theme management
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') as Theme;
    if (savedTheme) {
      setTheme(savedTheme);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('theme', theme);
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  // Input video preview URL management
  useEffect(() => {
    if (state.file) {
      const url = URL.createObjectURL(state.file);
      setInputVideoUrl(url);
      return () => URL.revokeObjectURL(url);
    }
    setInputVideoUrl(null);
  }, [state.file]);

  // Add this type at the top of the file with other types
  type Timeout = ReturnType<typeof setTimeout>;

  useEffect(() => {
    let interval: Timeout;
    let rotationInterval: Timeout;

    if (state.status === 'processing') {
      // Rotate the spinner continuously
      rotationInterval = setInterval(() => {
        setRotation(prev => (prev + 2) % 360);
      }, 16); // ~60fps

      // Change fun fact every 5 seconds
      interval = setInterval(() => {
        setCurrentFactIndex((prev) => (prev + 1) % VR_FACTS.length);
      }, 5000);
    }

    return () => {
      if (interval) clearInterval(interval);
      if (rotationInterval) clearInterval(rotationInterval);
    };
  }, [state.status]);

  useEffect(() => {
    if (state.status === 'processing') {
      const wsConnection = new WebSocket(`${WS_URL}/${clientId.current}`);
      
      wsConnection.onmessage = (event) => {
        const data = JSON.parse(event.data) as WebSocketMessage;
        setState(prev => ({
          ...prev,
          progress: data.progress,
          status: data.status === 'complete' ? 'success' : 'processing'
        }));
      };

      wsConnection.onclose = () => {
        console.log('WebSocket connection closed');
        wsRef.current = null;
      };

      wsRef.current = wsConnection;

      return () => {
        if (wsRef.current) {
          wsRef.current.close();
          wsRef.current = null;
        }
      };
    }
  }, [state.status]);

  const toggleTheme = () => {
    setTheme(theme === 'dark' ? 'light' : 'dark');
  };

  const handleReset = useCallback(() => {
    // Revoke existing object URLs to prevent memory leaks
    if (state.processedVideoUrl) {
      URL.revokeObjectURL(state.processedVideoUrl);
    }
    if (inputVideoUrl) {
      URL.revokeObjectURL(inputVideoUrl);
    }
    
    // Reset all state to initial values
    setState({
      file: null,
      status: 'idle',
      progress: 0,
      stereoLayout: 'side-by-side',
      quality: 'high',
      processedVideoUrl: null,
      error: null,
    });
    
    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    
    console.log('🔄 App state reset');
  }, [state.processedVideoUrl, inputVideoUrl]);

  const handleFileSelect = useCallback((file: File) => {
    if (!file.type.startsWith('video/')) {
      setState(prev => ({ ...prev, error: 'Please select a valid video file' }));
      return;
    }
    handleReset();
    setState(prev => ({
      ...prev,
      file,
      error: null,
    }));
  }, [handleReset]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  }, [handleFileSelect]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const handleConvert = async () => {
    if (!state.file) return;
    
    console.log('🎬 Starting conversion process...');
    setState(prev => ({ ...prev, error: null, status: 'processing', progress: 0 }));
    
    try {
      const response = await processVideo(
        {
          file: state.file,
          stereoLayout: state.stereoLayout,
          quality: state.quality
        },
        (progress) => {
          setState(prev => ({ ...prev, progress }));
        }
      );
      
      if (response.success && response.data) {
        const videoUrl = URL.createObjectURL(response.data);
        console.log('🎉 Conversion successful, created video URL');
        setState(prev => ({
          ...prev,
          status: 'success',
          processedVideoUrl: videoUrl,
          progress: 100,
        }));
      } else {
        console.error('💥 Conversion failed:', response.error);
        setState(prev => ({
          ...prev,
          status: 'error',
          error: response.error || 'Failed to process video',
        }));
      }
    } catch (error) {
      console.error('💥 Conversion error:', error);
      setState(prev => ({
        ...prev,
        status: 'error',
        error: error instanceof Error ? error.message : 'An unknown error occurred',
      }));
    }
  };

  const handleDownload = () => {
    if (state.processedVideoUrl) {
      const link = document.createElement('a');
      link.href = state.processedVideoUrl;
      link.download = state.file?.name ? `${state.file.name.split('.')[0]}_VR180.mp4` : 'converted_VR180.mp4';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      console.log('⬇️ Download initiated');
    }
  };

  const isProcessing = state.status === 'processing';

  const renderOutputState = () => {
    if (state.status === 'error' && state.error) {
      return (
        <div className="text-center text-red-400 space-y-4">
          <div className="w-16 h-16 mx-auto bg-red-500/20 rounded-full flex items-center justify-center">
            <X className="w-8 h-8 text-red-400" />
          </div>
          <h3 className="text-xl font-bold text-white mb-2">Processing Failed</h3>
          <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4">
            <p className="text-sm text-red-300">{state.error}</p>
          </div>
        </div>
      );
    }

    if (isProcessing) {
      return (
        <div className="text-center space-y-6">
          <div className="relative w-24 h-24 mx-auto">
            {/* Background circle */}
            <div className="absolute inset-0 rounded-full border-4 border-blue-500/20"></div>
            
            {/* Animated progress ring */}
            <div 
              className="absolute inset-0 rounded-full border-4 border-blue-500 border-t-transparent"
              style={{
                transform: `rotate(${rotation}deg)`,
                transition: 'transform 0.1s linear'
              }}
            ></div>
            
            {/* Loading spinner */}
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-xs text-blue-300">
                {state.progress < 100 ? 'Processing...' : 'Complete'}
              </span>
            </div>
            
            {/* Pulsing icon */}
            <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
              <Sparkles className="w-8 h-8 text-blue-400 animate-pulse" />
            </div>
          </div>
          
          {/* Fun fact */}
          <div className="bg-blue-500/10 rounded-lg p-3 text-sm">
            <p className="text-blue-300">{VR_FACTS[currentFactIndex]}</p>
          </div>
          
          <div className="text-xs text-gray-400">
            Processing with AI depth estimation...
          </div>
        </div>
      );
    }
    
    if (state.status === 'success' && state.processedVideoUrl) {
      return (
        <div className="text-center space-y-4 w-full">
          <CheckCircle2 className="w-16 h-16 text-green-400 mx-auto" />
          <h3 className="text-2xl font-bold text-white">Conversion Complete!</h3>
          <p className="text-sm text-gray-300">Your VR180 video is ready to view and download</p>
          
          <div className="bg-black/50 rounded-xl p-2">
            <video
              key={state.processedVideoUrl}
              src={state.processedVideoUrl}
              controls
              playsInline
              crossOrigin="anonymous"
              className="w-full rounded-lg shadow-lg bg-black"
              preload="metadata"
              autoPlay={false}
              controlsList="nodownload"
              onLoadedMetadata={() => console.log('Video metadata loaded')}
              onLoadedData={() => console.log('Video data loaded successfully')}
              onError={(e) => {
                console.error('Video Error:', e.currentTarget.error);
                console.error('Error code:', e.currentTarget.error?.code);
                console.error('Error message:', e.currentTarget.error?.message);
              }}
              onStalled={() => console.warn('Video playback stalled')}
              style={{ maxHeight: '300px' }}
            />
          </div>
          
          <div className="space-y-2">
            <button
              onClick={handleDownload}
              className="w-full py-3 px-6 rounded-xl bg-gradient-to-r from-green-500 to-emerald-600 text-white font-semibold text-base hover:from-green-400 hover:to-emerald-500 transform hover:scale-[1.02] transition-all duration-300 shadow-lg hover:shadow-xl"
            >
              <Download className="w-5 h-5 inline mr-2" />
              Download VR180 Video
            </button>
            
            <div className="text-xs text-gray-400 bg-white/5 rounded-lg p-2">
              <p><strong>Format:</strong> VR180 ({state.stereoLayout === 'side-by-side' ? 'Side-by-Side' : 'Top-Bottom'})</p>
              <p><strong>Quality:</strong> {state.quality.charAt(0).toUpperCase() + state.quality.slice(1)}</p>
            </div>
          </div>
        </div>
      );
    }
    
    // Idle state
    return (
      <div className="text-center text-gray-400">
        <Film className="w-16 h-16 mx-auto mb-4 opacity-30" />
        <h3 className="text-xl font-bold text-white">Output Preview</h3>
        <p>Your converted VR180 video will appear here</p>
        <p className="text-sm mt-2 text-gray-500">Complete 3D immersive experience with audio</p>
      </div>
    );
  };

  return (
    <div className="min-h-screen font-inter bg-gray-100 dark:bg-black transition-all duration-500">
      <div className="absolute inset-0 bg-gradient-to-br from-gray-900 via-black to-blue-900 opacity-80 dark:opacity-100"></div>
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(56,189,248,0.1),transparent_50%)]"></div>
      
      <div className="absolute top-6 right-6 z-20">
        <button
          onClick={toggleTheme}
          className="p-3 rounded-xl bg-white/10 backdrop-blur-sm border border-white/20 hover:bg-white/20 transition-all duration-300 group"
        >
          {theme === 'dark' ? (
            <Sun className="w-5 h-5 text-yellow-400 group-hover:rotate-12 transition-transform duration-300" />
          ) : (
            <Moon className="w-5 h-5 text-blue-600 group-hover:rotate-12 transition-transform duration-300" />
          )}
        </button>
      </div>

      <div className="relative z-10 min-h-screen flex flex-col items-center justify-center p-4">
        <div className="text-center mb-8 animate-fade-in">
          <h1 className="font-orbitron text-4xl md:text-6xl font-black bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 bg-clip-text text-transparent mb-3">
            STELLAR VR
          </h1>
          <p className="text-base md:text-lg text-gray-300 flex items-center justify-center gap-2">
            <Sparkles className="w-4 h-4 text-cyan-400" />
            AI-Powered 2D to VR180 Converter
            <Sparkles className="w-4 h-4 text-cyan-400" />
          </p>
        </div>

        <div className="w-full max-w-5xl bg-white/5 backdrop-blur-xl rounded-2xl border border-white/10 shadow-xl p-6 md:p-8 animate-slide-up">
          <div className="grid md:grid-cols-2 gap-8">
            
            {/* Input Section */}
            <div className="space-y-6">
              <h2 className="text-xl md:text-2xl font-bold text-white text-center">
                Input Video
              </h2>
              
              {!state.file ? (
                <div
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  onClick={() => fileInputRef.current?.click()}
                  className="border-2 border-dashed border-cyan-400/30 rounded-xl p-8 md:p-10 text-center cursor-pointer hover:border-cyan-400/60 hover:bg-white/5 transition-all duration-300 group"
                >
                  <Upload className="w-12 h-12 text-cyan-400/70 mx-auto mb-4 group-hover:scale-110 group-hover:text-cyan-400 transition-all duration-300" />
                  <p className="text-lg md:text-xl font-semibold text-white mb-2">
                    Drag & Drop Your Video
                  </p>
                  <p className="text-sm text-gray-400">
                    or click to browse • MP4, MOV, AVI, MKV, WEBM
                  </p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="video/*"
                    onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
                    className="hidden"
                  />
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="bg-white/10 rounded-xl p-4 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className="p-2 rounded-lg bg-cyan-400/20">
                        <Film className="w-5 h-5 text-cyan-400" />
                      </div>
                      <div>
                        <p className="font-medium text-white text-sm">
                          {state.file.name}
                        </p>
                        <p className="text-xs text-gray-400">
                          {(state.file.size / 1024 / 1024).toFixed(1)} MB
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={handleReset}
                      className="p-2 rounded-lg bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors duration-200"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>

                  {inputVideoUrl && (
                    <div className="bg-black/30 rounded-xl p-2">
                      <video
                        src={inputVideoUrl}
                        controls
                        className="w-full rounded-lg shadow-lg bg-black"
                        preload="metadata"
                        style={{ maxHeight: '300px' }}
                      />
                    </div>
                  )}
                </div>
              )}
              
              {state.file && (
                <div className="space-y-4 pt-4 border-t border-white/10">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs font-medium text-gray-300 mb-2 uppercase tracking-wide">
                        Stereo Layout
                      </label>
                      <select
                        value={state.stereoLayout}
                        onChange={(e) => setState(prev => ({ ...prev, stereoLayout: e.target.value as StereoLayout }))}
                        className="w-full p-3 rounded-lg bg-white/10 border border-white/20 text-white text-sm focus:outline-none focus:ring-2 focus:ring-cyan-400/50 focus:border-transparent transition-all duration-200"
                        disabled={isProcessing}
                      >
                        <option value="side-by-side" className="bg-gray-800">Side by Side</option>
                        <option value="over-under" className="bg-gray-800">Top Bottom</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-300 mb-2 uppercase tracking-wide">
                        Quality
                      </label>
                      <select
                        value={state.quality}
                        onChange={(e) => setState(prev => ({ ...prev, quality: e.target.value as Quality }))}
                        className="w-full p-3 rounded-lg bg-white/10 border border-white/20 text-white text-sm focus:outline-none focus:ring-2 focus:ring-cyan-400/50 focus:border-transparent transition-all duration-200"
                        disabled={isProcessing}
                      >
                        <option value="high" className="bg-gray-800">High</option>
                        <option value="medium" className="bg-gray-800">Medium</option>
                        <option value="low" className="bg-gray-800">Low</option>
                      </select>
                    </div>
                  </div>
                  <button
                    onClick={handleConvert}
                    disabled={!state.file || isProcessing}
                    className="w-full py-3 px-6 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold text-base hover:from-cyan-400 hover:to-blue-500 transform hover:scale-[1.02] transition-all duration-300 shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
                  >
                    <Rocket className="w-5 h-5 inline mr-2" />
                    {isProcessing ? 'Converting...' : 'Convert to VR180'}
                  </button>
                </div>
              )}
            </div>

            {/* Output Section */}
            <div className="space-y-6 flex flex-col items-center justify-center bg-white/5 rounded-xl p-6 min-h-[400px] md:min-h-full">
              {renderOutputState()}
              {(state.status === 'success' || state.status === 'error') && (
                <button
                  onClick={handleReset}
                  className="w-full mt-4 py-2.5 px-5 rounded-lg border border-cyan-400/50 text-cyan-400 font-medium text-sm hover:bg-cyan-400/10 hover:border-cyan-400 transition-all duration-300"
                >
                  Convert Another Video
                </button>
              )}
            </div>
          </div>
        </div>

        <div className="text-center mt-6 text-gray-500 space-y-1">
          <p className="flex items-center justify-center gap-2 text-sm">
            <Zap className="w-4 h-4" />
            Powered by Advanced AI Technology
          </p>
          <p className="text-xs">© 2024 STELLAR VR. All rights reserved.</p>
        </div>
      </div>
    </div>
  );
}

export default App;