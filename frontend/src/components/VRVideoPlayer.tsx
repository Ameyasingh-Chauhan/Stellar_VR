import React, { useRef, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Play, Pause, Volume2, VolumeX, Maximize2 } from 'lucide-react';

interface VRVideoPlayerProps {
  videoUrl: string;
  className?: string;
}

export const VRVideoPlayer: React.FC<VRVideoPlayerProps> = ({ 
  videoUrl, 
  className = '' 
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [showControls, setShowControls] = useState(true);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [rotation, setRotation] = useState({ x: 0, y: 0 });

  useEffect(() => {
    let timeoutId: ReturnType<typeof setTimeout>;
    
    const handleMouseMove = (e: MouseEvent) => {
      if (!containerRef.current) return;
      
      const rect = containerRef.current.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width;
      const y = (e.clientY - rect.top) / rect.height;
      
      setMousePos({ x, y });
      
      if (isDragging) {
        // Create VR-like look-around effect
        const deltaX = (x - 0.5) * 60; // 60 degrees max rotation
        const deltaY = (y - 0.5) * 30; // 30 degrees max rotation
        setRotation({ x: deltaY, y: deltaX });
      }
      
      setShowControls(true);
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => setShowControls(false), 3000);
    };

    const container = containerRef.current;
    if (container) {
      container.addEventListener('mousemove', handleMouseMove);
      return () => {
        container.removeEventListener('mousemove', handleMouseMove);
        clearTimeout(timeoutId);
      };
    }
  }, [isDragging]);

  const togglePlayback = () => {
    if (videoRef.current) {
      if (videoRef.current.paused) {
        videoRef.current.play();
        setIsPlaying(true);
      } else {
        videoRef.current.pause();
        setIsPlaying(false);
      }
    }
  };

  const toggleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !videoRef.current.muted;
      setIsMuted(videoRef.current.muted);
    }
  };

  const enterFullscreen = () => {
    if (videoRef.current) {
      if (videoRef.current.requestFullscreen) {
        videoRef.current.requestFullscreen();
      }
    }
  };

  return (
    <div 
      ref={containerRef}
      className={`relative bg-black rounded-xl overflow-hidden group cursor-grab ${isDragging ? 'cursor-grabbing' : ''} ${className}`}
      onMouseDown={() => setIsDragging(true)}
      onMouseUp={() => setIsDragging(false)}
      onMouseLeave={() => setIsDragging(false)}
    >
      {/* Magic Window Effect */}
      <div 
        className="relative overflow-hidden"
        style={{
          transform: `perspective(1000px) rotateX(${rotation.x}deg) rotateY(${rotation.y}deg)`,
          transition: isDragging ? 'none' : 'transform 0.3s ease-out'
        }}
      >
        <video
          ref={videoRef}
          src={videoUrl}
          className="w-full h-auto max-h-80 object-cover"
          loop
          playsInline
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
        />
        
        {/* VR Overlay Effect */}
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-transparent to-transparent opacity-20 pointer-events-none">
          <div 
            className="w-full h-full bg-gradient-radial from-transparent to-black/20"
            style={{
              background: `radial-gradient(circle at ${mousePos.x * 100}% ${mousePos.y * 100}%, transparent 30%, rgba(0,0,0,0.1) 70%)`
            }}
          />
        </div>
      </div>

      {/* Custom Controls Overlay */}
      <motion.div
        initial={{ opacity: 1 }}
        animate={{ opacity: showControls ? 1 : 0 }}
        transition={{ duration: 0.3 }}
        className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent flex items-end justify-between p-4 pointer-events-none"
      >
        <div className="flex items-center gap-3 pointer-events-auto">
          <motion.button
            onClick={togglePlayback}
            className="w-10 h-10 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center hover:bg-white/30 transition-colors"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {isPlaying ? (
              <Pause className="w-5 h-5 text-white" />
            ) : (
              <Play className="w-5 h-5 text-white ml-0.5" />
            )}
          </motion.button>
          
          <motion.button
            onClick={toggleMute}
            className="w-8 h-8 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center hover:bg-white/30 transition-colors"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {isMuted ? (
              <VolumeX className="w-4 h-4 text-white" />
            ) : (
              <Volume2 className="w-4 h-4 text-white" />
            )}
          </motion.button>
        </div>
        
        <motion.button
          onClick={enterFullscreen}
          className="w-8 h-8 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center hover:bg-white/30 transition-colors pointer-events-auto"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <Maximize2 className="w-4 h-4 text-white" />
        </motion.button>
      </motion.div>

      {/* VR Instructions */}
      <div className="absolute top-4 left-4 right-4">
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: showControls ? 1 : 0, y: showControls ? 0 : -10 }}
          className="bg-black/60 backdrop-blur-sm rounded-lg px-3 py-2 text-center pointer-events-none"
        >
          <p className="text-white text-xs font-medium">
            🕶️ Magic Window: Drag to look around in VR
          </p>
        </motion.div>
      </div>

      {/* Play button overlay for initial state */}
      {!isPlaying && (
        <motion.div
          className="absolute inset-0 flex items-center justify-center bg-black/30 cursor-pointer"
          onClick={togglePlayback}
          whileHover={{ backgroundColor: 'rgba(0,0,0,0.4)' }}
        >
          <motion.div
            className="w-16 h-16 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center"
            whileHover={{ scale: 1.1, backgroundColor: 'rgba(255,255,255,0.3)' }}
            whileTap={{ scale: 0.9 }}
          >
            <Play className="w-8 h-8 text-white ml-1" />
          </motion.div>
        </motion.div>
      )}
    </div>
  );
};

export default VRVideoPlayer;