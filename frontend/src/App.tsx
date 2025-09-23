import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, X, CheckCircle2, Download, Film, Sun, Moon, Upload, Rocket, Zap, Play, Pause, RotateCcw, Share2 } from 'lucide-react';

// Components
import { Login } from '@/components/Login';
import { UserProfile } from '@/components/UserProfile';
import { Sidebar } from '@/components/Sidebar';
import { About } from '@/components/About';
import { ImmersionSlider } from '@/components/ImmersionSlider';
import { ProcessingTimeline } from '@/components/ProcessingTimeline';
import { VRVideoPlayer } from '@/components/VRVideoPlayer';
import { QRCodeDisplay } from '@/components/QRCodeDisplay';

// Backend configuration - DO NOT CHANGE
const API_URL = 'http://127.0.0.1:8000';
const WS_URL = 'ws://127.0.0.1:8001/ws';

interface ProcessVideoParams {
  file: File;
  stereoLayout: 'side-by-side' | 'over-under';
  quality: 'high' | 'medium' | 'low';
  useNewPipeline: boolean;
}

interface ProcessVideoResponse {
  success: boolean;
  data?: Blob;
  error?: string;
  sessionId?: string;
}

/** 
 * Sends the video file and options to the backend for processing.
 * CRITICAL: DO NOT MODIFY THIS FUNCTION
 */
const processVideo = async (params: ProcessVideoParams, onProgress?: (progress: number, message?: string) => void): Promise<ProcessVideoResponse> => {
  const { file, stereoLayout, quality, useNewPipeline } = params;

  const formData = new FormData();
  formData.append('file', file);
  formData.append('stereo_layout', stereoLayout === 'side-by-side' ? 'left_right' : 'top_bottom');
  formData.append('quality', quality);
  formData.append('use_new_pipeline', useNewPipeline.toString());

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 18000000);

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

    const sessionId = response.headers.get('X-Session-ID') || undefined;
    
    const contentType = response.headers.get('content-type');
    if (!contentType?.includes('video/') && !contentType?.includes('application/octet-stream')) {
      throw new Error('Server returned invalid content type');
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body available');
    }

    const chunks: BlobPart[] = [];
    let receivedLength = 0;
    const contentLength = parseInt(response.headers.get('content-length') || '0', 10);

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      receivedLength += value.length;
      
      if (contentLength && onProgress) {
        const percentComplete = Math.min(95 + ((receivedLength / contentLength) * 5), 99);
        onProgress(percentComplete, 'Finalizing video...');
      }
    }

    const blob = new Blob(chunks, { type: 'video/mp4' });
    
    if (!blob || blob.size === 0) {
      throw new Error('Received empty video response');
    }

    return { success: true, data: blob, sessionId };
  } catch (error: unknown) {
    console.error("❌ Network or fetch error:", error);
    return { 
      success: false, 
      error: error instanceof Error ? error.message : 'Unknown error'
    };
  }
};

// Component Types
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
  useNewPipeline: boolean; // Always true now
  processedVideoUrl: string | null;
  error: string | null;
  progressMessage: string | null;
  sessionId: string | null;
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
  "🎭 Round-2 pipeline adds cinematic dome effects",
  "💫 Foveated rendering reduces eye fatigue by 40%",
  "🔮 Depth-aware inpainting fills occlusion holes naturally",
  "🌐 Periphery expansion creates true 210° field of view"
];

const App: React.FC = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [theme, setTheme] = useState<Theme>('dark');
  const [currentView, setCurrentView] = useState<'home' | 'about'>('home');
  const [state, setState] = useState<AppState>({
    file: null,
    status: 'idle',
    progress: 0,
    stereoLayout: 'side-by-side',
    quality: 'high',
    useNewPipeline: true, // Hardcoded to true
    processedVideoUrl: null,
    error: null,
    progressMessage: null,
    sessionId: null
  });
  const [inputVideoUrl, setInputVideoUrl] = useState<string | null>(null);
  const [currentFactIndex, setCurrentFactIndex] = useState(0);
  const [isPlayingPreview, setIsPlayingPreview] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const inputVideoRef = useRef<HTMLVideoElement>(null);

  // WebSocket connection management - DO NOT MODIFY
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

  // Fun fact carousel
  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | null = null;

    if (state.status === 'processing') {
      interval = setInterval(() => {
        setCurrentFactIndex((prev) => (prev + 1) % VR_FACTS.length);
      }, 5000);
    }

    return () => {
      if (interval) clearInterval(interval);
    }
  }, [state.status]);

  // WebSocket for progress updates - DO NOT MODIFY
  useEffect(() => {
    if (state.status === 'processing') {
      const wsUrl = `${WS_URL}/${clientId.current}`;
      console.log(`🔌 Connecting to WebSocket: ${wsUrl}`);
      
      const wsConnection = new WebSocket(wsUrl);
      
      wsConnection.onopen = () => {
        console.log('✅ WebSocket connection opened');
      };
      
      wsConnection.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('📡 WebSocket message:', data);
          
          if (data.progress !== undefined) {
            setState(prev => ({
              ...prev,
              progress: data.progress,
              progressMessage: data.message || 'Processing...'
            }));
          }
          
          if (data.status === 'completed') {
            setState(prev => ({ ...prev, status: 'success' }));
          }
        } catch (error: unknown) {
          const errMsg = error instanceof Error ? error.message : String(error);
          setState(prev => ({
            ...prev,
            error: errMsg,
            status: 'error'
          }));
        }
      };

      wsConnection.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
      };

      wsConnection.onclose = () => {
        console.log('🔌 WebSocket connection closed');
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
    if (state.processedVideoUrl) {
      URL.revokeObjectURL(state.processedVideoUrl);
    }
    if (inputVideoUrl) {
      URL.revokeObjectURL(inputVideoUrl);
    }
    
    setState({
      file: null,
      status: 'idle',
      progress: 0,
      stereoLayout: 'side-by-side',
      quality: 'high',
      useNewPipeline: true,
      processedVideoUrl: null,
      error: null,
      progressMessage: null,
      sessionId: null
    });
    
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
    setState(prev => ({ 
      ...prev, 
      error: null, 
      status: 'processing', 
      progress: 0,
      progressMessage: 'Starting conversion...'
    }));
    
    // Create a controller to potentially cancel the dummy loading
    const controller = new AbortController();
    const { signal } = controller;
    
    // Start dummy loading with minimum 10 minutes
    const startTime = Date.now();
    const minDuration = 10 * 60 * 1000; // 10 minutes in milliseconds
    const maxProgress = 98; // Stop at 98% if taking longer than 10 minutes
    
    // Function to update progress with dummy loading
    const updateProgress = (actualProgress: number, message?: string) => {
      const elapsedTime = Date.now() - startTime;
      const isMinTimeElapsed = elapsedTime >= minDuration;
      
      // If minimum time has elapsed, cap progress at 98%
      const progress = isMinTimeElapsed ? Math.min(actualProgress, maxProgress) : actualProgress;
      
      setState(prev => ({ 
        ...prev, 
        progress,
        progressMessage: message || `Processing... ${Math.round(progress)}%`
      }));
      
      // If minimum time elapsed and actual progress is 100%, complete the process
      if (isMinTimeElapsed && actualProgress >= 100) {
        setState(prev => ({
          ...prev,
          status: 'success',
          progress: 100,
          progressMessage: 'Conversion completed!'
        }));
      }
      
      return { isMinTimeElapsed, progress };
    };
    
    try {
      // Start the dummy timer
      const dummyTimer = setInterval(() => {
        if (signal.aborted) {
          clearInterval(dummyTimer);
          return;
        }
        
        const elapsedTime = Date.now() - startTime;
        const isMinTimeElapsed = elapsedTime >= minDuration;
        
        // Only update if we haven't reached the max progress yet
        if (state.progress < maxProgress) {
          // Calculate dummy progress (0-98% over 10 minutes)
          const dummyProgress = Math.min((elapsedTime / minDuration) * maxProgress, maxProgress);
          setState(prev => ({ 
            ...prev, 
            progress: dummyProgress,
            progressMessage: `Processing... ${Math.round(dummyProgress)}%`
          }));
        }
        
        // If minimum time has elapsed, we can finish
        if (isMinTimeElapsed) {
          clearInterval(dummyTimer);
        }
      }, 1000); // Update every second
      
      const response = await processVideo(
        {
          file: state.file,
          stereoLayout: state.stereoLayout,
          quality: state.quality,
          useNewPipeline: state.useNewPipeline
        },
        (progress, message) => {
          if (signal.aborted) return;
          
          const { isMinTimeElapsed } = updateProgress(progress, message);
          
          // If minimum time has elapsed and we have actual progress data, we can proceed
          if (isMinTimeElapsed) {
            // Clear the dummy timer
            clearInterval(dummyTimer);
          }
        }
      );
      
      // Clear the dummy timer
      clearInterval(dummyTimer);
      
      // Abort any further dummy loading
      controller.abort();
      
      if (response.success && response.data) {
        const videoUrl = URL.createObjectURL(response.data);
        console.log('🎉 Conversion successful, created video URL');
        setState(prev => ({
          ...prev,
          status: 'success',
          processedVideoUrl: videoUrl,
          progress: 100,
          progressMessage: 'Conversion completed!',
          sessionId: response.sessionId || null
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
      // Clear the dummy timer in case of error
      controller.abort();
      
      console.error('💥 Conversion error:', error);
      setState(prev => ({
        ...prev,
        status: 'error',
        error: error instanceof Error ? error.message : 'An unknown error occurred',
      }));
    }
  };

  const handleDownload = () => {
    if (state.sessionId) {
      const downloadUrl = `${API_URL}/download/${state.sessionId}`;
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = state.file?.name ? `${state.file.name.split('.')[0]}_VR180_HQ.mp4` : 'converted_VR180_HQ.mp4';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      console.log('⬇️ High-quality download initiated');
    } else if (state.processedVideoUrl) {
      const link = document.createElement('a');
      link.href = state.processedVideoUrl;
      link.download = state.file?.name ? `${state.file.name.split('.')[0]}_VR180.mp4` : 'converted_VR180.mp4';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      console.log('⬇️ Download initiated');
    }
  };

  const handleShare = async () => {
    if (state.processedVideoUrl) {
      try {
        await navigator.clipboard.writeText(state.processedVideoUrl);
        // Could add toast notification here
      } catch (err) {
        console.error('Failed to copy to clipboard:', err);
      }
    }
  };

  const togglePreviewPlayback = () => {
    const video = inputVideoRef.current;
    if (video) {
      if (video.paused) {
        video.play();
        setIsPlayingPreview(true);
      } else {
        video.pause();
        setIsPlayingPreview(false);
      }
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const isProcessing = state.status === 'processing';

  const handleLogin = (success: boolean) => {
    if (success) {
      setIsLoggedIn(true);
    }
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    // Reset any user-specific state
    handleReset();
  };

  // Show login page if not logged in
  if (!isLoggedIn) {
    return <Login onLogin={handleLogin} />;
  }

  return (
    <div className="min-h-screen font-sans bg-gradient-to-br from-background via-background to-muted text-foreground overflow-hidden">
      {/* Subtle background animation */}
      <div className="constellation-bg"></div>
      
      {/* Main Layout */}
      <div className="flex min-h-screen relative z-10">
        {/* Sidebar */}
        <Sidebar activeView={currentView} onViewChange={setCurrentView} />

        {/* Main Content */}
        <div className="flex-1 flex flex-col">
          {/* Header */}
          <header className="bg-card/50 backdrop-blur-sm border-b border-border py-4 px-6 flex items-center justify-between">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center gap-3"
            >
              <h1 className="text-2xl font-bold bg-gradient-to-r from-vr-primary to-vr-secondary bg-clip-text text-transparent">
                STELLAR VR
              </h1>
              <span className="text-sm text-muted-foreground">
                {currentView === 'home' ? 'Conversion Hub' : 'About the Tech'}
              </span>
            </motion.div>
            
            <div className="flex items-center gap-4">
              <UserProfile 
                onLogout={handleLogout}
                theme={theme}
                onThemeToggle={toggleTheme}
              />
            </div>
          </header>

          {/* Content Area */}
          <main className="flex-1 p-6">
            <AnimatePresence mode="wait">
              {currentView === 'about' ? (
                <motion.div
                  key="about"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <About />
                </motion.div>
              ) : (
                <motion.div
                  key="home"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  transition={{ duration: 0.3 }}
                  className="max-w-7xl mx-auto"
                >
                  {/* Hero Section */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                    className="text-center mb-12"
                  >
                    <h1 className="text-4xl md:text-5xl font-bold mb-4">
                      Transform Videos into{" "}
                      <span className="bg-gradient-to-r from-vr-primary to-vr-secondary bg-clip-text text-transparent">
                        Immersive VR
                      </span>
                    </h1>
                    <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
                      Convert any 2D video into stunning 180° VR experiences with AI-powered depth perception
                    </p>
                  </motion.div>

                  <div className="grid lg:grid-cols-2 gap-8">
                    {/* Input Section */}
                    <motion.div
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.5 }}
                      className="bg-card border border-border rounded-xl overflow-hidden shadow-lg"
                    >
                      <div className="p-6 border-b border-border">
                        <div className="flex items-center justify-between">
                          <h2 className="text-xl font-bold flex items-center gap-2">
                            <Upload className="w-5 h-5 text-vr-primary" />
                            Input Video
                          </h2>
                          {state.file && (
                            <button
                              onClick={handleReset}
                              className="text-muted-foreground hover:text-foreground transition-colors"
                            >
                              <RotateCcw className="w-5 h-5" />
                            </button>
                          )}
                        </div>
                      </div>
                      
                      <div className="p-6">
                        {!state.file ? (
                          <motion.div
                            onDrop={handleDrop}
                            onDragOver={handleDragOver}
                            onClick={() => fileInputRef.current?.click()}
                            className="border-2 border-dashed border-vr-primary/30 rounded-xl p-8 text-center cursor-pointer hover:border-vr-primary/60 hover:bg-vr-primary/5 transition-all duration-300 group"
                            whileHover={{ scale: 1.02 }}
                          >
                            <Upload className="w-12 h-12 text-vr-primary/70 mx-auto mb-4 group-hover:scale-110 group-hover:text-vr-primary transition-all duration-300" />
                            <h3 className="text-lg font-semibold text-foreground mb-2">
                              Drag & Drop Your Video
                            </h3>
                            <p className="text-muted-foreground text-sm">
                              or click to browse • MP4, MOV, AVI, MKV, WEBM
                            </p>
                            <input
                              ref={fileInputRef}
                              type="file"
                              accept="video/*"
                              onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
                              className="hidden"
                            />
                          </motion.div>
                        ) : (
                          <motion.div
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="space-y-6"
                          >
                            <div className="bg-muted/50 rounded-xl p-4">
                              <div className="flex items-start gap-3">
                                <div className="p-2 rounded-lg bg-vr-primary/20">
                                  <Film className="w-5 h-5 text-vr-primary" />
                                </div>
                                <div className="flex-1 min-w-0">
                                  <p className="font-medium text-foreground text-sm truncate">
                                    {state.file.name}
                                  </p>
                                  <p className="text-xs text-muted-foreground mt-1">
                                    {formatFileSize(state.file.size)} • {state.file.type.split('/')[1]?.toUpperCase()}
                                  </p>
                                </div>
                              </div>
                            </div>

                            {inputVideoUrl && (
                              <div className="bg-black/50 rounded-xl p-3 border border-border">
                                <div className="relative group">
                                  <video
                                    ref={inputVideoRef}
                                    src={inputVideoUrl}
                                    className="w-full rounded-lg shadow-lg bg-black"
                                    preload="metadata"
                                    style={{ maxHeight: '200px' }}
                                  />
                                  <div className="absolute inset-0 bg-black/30 rounded-lg flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
                                       onClick={togglePreviewPlayback}>
                                    {isPlayingPreview ? (
                                      <Pause className="w-10 h-10 text-white/80" />
                                    ) : (
                                      <Play className="w-10 h-10 text-white/80" />
                                    )}
                                  </div>
                                </div>
                              </div>
                            )}

                            {/* Immersion Level Slider */}
                            <div className="space-y-6 pt-6 border-t border-border">
                              <ImmersionSlider
                                value={state.quality}
                                onChange={(quality) => setState(prev => ({ ...prev, quality }))}
                                disabled={isProcessing}
                              />

                              <motion.button
                                onClick={handleConvert}
                                disabled={!state.file || isProcessing}
                                className="w-full py-4 px-6 rounded-xl bg-gradient-to-r from-vr-primary to-vr-secondary text-white font-semibold text-base hover:shadow-lg hover:shadow-vr-primary/25 transform hover:scale-[1.02] transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 flex items-center justify-center"
                                whileHover={!isProcessing ? { scale: 1.02 } : {}}
                                whileTap={!isProcessing ? { scale: 0.98 } : {}}
                              >
                                <Rocket className="w-5 h-5 inline mr-2" />
                                {isProcessing ? 'Converting...' : 'Convert to VR180'}
                              </motion.button>
                            </div>
                          </motion.div>
                        )}
                      </div>
                    </motion.div>

                    {/* Output Section */}
                    <motion.div
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.5, delay: 0.1 }}
                      className="bg-card border border-border rounded-xl overflow-hidden shadow-lg"
                    >
                      <div className="p-6 border-b border-border">
                        <h2 className="text-xl font-bold flex items-center gap-2">
                          <Film className="w-5 h-5 text-vr-success" />
                          Output VR180
                        </h2>
                      </div>
                      
                      <div className="p-6 min-h-[500px] flex flex-col">
                        <div className="flex-1 flex flex-col justify-center">
                          <AnimatePresence mode="wait">
                            {state.status === 'error' && state.error && (
                              <motion.div
                                key="error"
                                initial={{ opacity: 0, scale: 0.9 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.9 }}
                                className="text-center space-y-6"
                              >
                                <div className="w-20 h-20 mx-auto bg-red-500/20 rounded-full flex items-center justify-center">
                                  <X className="w-10 h-10 text-red-500" />
                                </div>
                                <div>
                                  <h3 className="text-2xl font-bold text-foreground mb-2">Processing Failed</h3>
                                  <p className="text-muted-foreground">Something went wrong during the conversion</p>
                                </div>
                                <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-5 text-left">
                                  <p className="text-red-400 font-medium">Error Details</p>
                                  <p className="text-sm text-red-300 mt-2">{state.error}</p>
                                </div>
                                <button
                                  onClick={handleReset}
                                  className="mt-4 py-3 px-6 rounded-xl bg-gradient-to-r from-red-600 to-red-700 text-white font-semibold hover:from-red-500 hover:to-red-600 transition-all duration-300 shadow-lg hover:shadow-xl"
                                >
                                  <RotateCcw className="w-5 h-5 inline mr-2" />
                                  Try Again
                                </button>
                              </motion.div>
                            )}

                            {isProcessing && (
                              <motion.div
                                key="processing"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                className="text-center space-y-6 w-full"
                              >
                                {/* Smooth CSS Spinner */}
                                <div className="relative w-32 h-32 mx-auto mb-8">
                                  <div className="absolute inset-0 rounded-full border-4 border-vr-primary/20"></div>
                                  <div className="absolute inset-0 rounded-full border-4 border-vr-primary border-t-transparent animate-spin-smooth"></div>
                                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                                    <span className="text-2xl font-bold text-vr-primary">
                                      {Math.round(state.progress)}%
                                    </span>
                                  </div>
                                </div>

                                {/* Processing Timeline */}
                                <ProcessingTimeline
                                  progress={state.progress}
                                  currentMessage={state.progressMessage}
                                />

                                {/* Fun fact carousel */}
                                <motion.div
                                  key={currentFactIndex}
                                  initial={{ opacity: 0, y: 20 }}
                                  animate={{ opacity: 1, y: 0 }}
                                  exit={{ opacity: 0, y: -20 }}
                                  className="bg-gradient-to-r from-vr-secondary/20 to-vr-primary/20 rounded-xl p-5 border border-vr-primary/20"
                                >
                                  <div className="flex items-start gap-3">
                                    <Sparkles className="w-5 h-5 text-vr-primary mt-0.5 flex-shrink-0" />
                                    <div className="text-left">
                                      <p className="text-vr-primary text-sm font-medium">Did you know?</p>
                                      <p className="text-foreground text-sm mt-1">{VR_FACTS[currentFactIndex]}</p>
                                    </div>
                                  </div>
                                </motion.div>
                                
                                <div className="text-xs text-muted-foreground flex items-center justify-center gap-2">
                                  <Zap className="w-4 h-4" />
                                  <span>Powered by Advanced AI Technology</span>
                                </div>
                              </motion.div>
                            )}

                            {state.status === 'success' && state.processedVideoUrl && (
                              <motion.div
                                key="success"
                                initial={{ opacity: 0, scale: 0.9 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.9 }}
                                className="text-center space-y-6 w-full"
                              >
                                <div className="w-16 h-16 mx-auto bg-vr-success/20 rounded-full flex items-center justify-center">
                                  <CheckCircle2 className="w-8 h-8 text-vr-success" />
                                </div>
                                <div>
                                  <h3 className="text-2xl font-bold text-foreground">Conversion Complete!</h3>
                                  <p className="text-muted-foreground mt-1">Your VR180 video is ready to view and download</p>
                                </div>
                                
                                {/* VR Video Player with Magic Window */}
                                <VRVideoPlayer 
                                  videoUrl={state.processedVideoUrl}
                                  className="mb-6"
                                />

                                {/* Video Info Grid */}
                                <div className="grid grid-cols-2 gap-4 mb-6">
                                  <div className="bg-muted/50 rounded-lg p-3 text-left">
                                    <p className="text-xs text-muted-foreground">Format</p>
                                    <p className="text-foreground font-medium text-sm">VR180 ({state.stereoLayout === 'side-by-side' ? 'SBS' : 'TB'})</p>
                                  </div>
                                  <div className="bg-muted/50 rounded-lg p-3 text-left">
                                    <p className="text-xs text-muted-foreground">Quality</p>
                                    <p className="text-foreground font-medium text-sm">{state.quality.charAt(0).toUpperCase() + state.quality.slice(1)}</p>
                                  </div>
                                </div>


                                {/* Action Buttons */}
                                <div className="grid grid-cols-2 gap-4">
                                  <button
                                    onClick={handleDownload}
                                    className="py-3 px-4 rounded-xl bg-gradient-to-r from-vr-success to-emerald-600 text-white font-semibold hover:shadow-lg hover:shadow-vr-success/25 transform hover:scale-[1.02] transition-all duration-300 flex items-center justify-center"
                                  >
                                    <Download className="w-4 h-4 inline mr-2" />
                                    Download
                                  </button>
                                  
                                </div>
                              </motion.div>
                            )}

                            {state.status === 'idle' && (
                              <motion.div
                                key="idle"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                className="text-center space-y-4"
                              >
                                <div className="w-16 h-16 mx-auto bg-muted/50 rounded-full flex items-center justify-center">
                                  <Film className="w-8 h-8 text-muted-foreground" />
                                </div>
                                <div>
                                  <h3 className="text-xl font-bold text-foreground">Output Preview</h3>
                                  <p className="text-muted-foreground text-sm mt-1">Your converted VR180 video will appear here</p>
                                </div>
                                <div className="bg-vr-primary/5 border border-vr-primary/20 rounded-lg p-4 mt-4">
                                  <p className="text-muted-foreground text-sm">
                                    <span className="font-medium">Tip:</span> The pipeline adds cinematic dome effects and foveated rendering for the ultimate VR experience.
                                  </p>
                                </div>
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </div>
                        
                        {(state.status === 'success' || state.status === 'error') && (
                          <motion.button
                            onClick={handleReset}
                            className="w-full mt-6 py-3 px-5 rounded-lg border border-vr-primary/50 text-vr-primary font-medium hover:bg-vr-primary/10 hover:border-vr-primary transition-all duration-300 flex items-center justify-center gap-2"
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                          >
                            <RotateCcw className="w-4 h-4" />
                            Convert Another Video
                          </motion.button>
                        )}
                      </div>
                    </motion.div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </main>
        </div>
      </div>
    </div>
  );
};

export default App;