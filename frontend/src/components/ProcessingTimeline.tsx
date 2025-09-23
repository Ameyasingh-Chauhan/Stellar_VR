import React from 'react';
import { motion } from 'framer-motion';
import { 
  Upload, 
  Eye, 
  Brain, 
  Layers, 
  Sparkles,
  CheckCircle2,
  Circle,
  Loader2
} from 'lucide-react';

interface ProcessingStep {
  id: string;
  label: string;
  icon: React.ComponentType<React.SVGProps<SVGSVGElement>>;
  keywords: string[];
}

interface ProcessingTimelineProps {
  progress: number;
  currentMessage: string | null;
}

const steps: ProcessingStep[] = [
  {
    id: 'upload',
    label: 'Video Uploaded',
    icon: Upload,
    keywords: ['upload', 'file', 'received']
  },
  {
    id: 'motion',
    label: 'Analyzing Motion', 
    icon: Eye,
    keywords: ['motion', 'analyzing', 'analysis', 'tracking']
  },
  {
    id: 'depth',
    label: 'Generating Depth Map',
    icon: Brain,
    keywords: ['depth', 'map', 'midas', 'generating', 'estimating']
  },
  {
    id: 'render',
    label: 'Rendering VR Views',
    icon: Layers,
    keywords: ['render', 'stereo', 'vr', 'views', 'processing', 'dibr', 'outpaint', 'frame']
  },
  {
    id: 'finalize',
    label: 'Finalizing Experience',
    icon: Sparkles,
    keywords: ['final', 'complete', 'finished', 'encoding', 'packaging']
  }
];

export const ProcessingTimeline: React.FC<ProcessingTimelineProps> = ({
  progress,
  currentMessage
}) => {
  const getCurrentStep = () => {
    const message = currentMessage?.toLowerCase() || '';
    
    // Special handling for frame processing messages
    if (message.includes('frame') && message.includes('/')) {
      return 'render';
    }
    
    // Find step based on keywords in the message
    const foundStep = steps.find(step => 
      step.keywords.some(keyword => message.includes(keyword))
    );
    
    if (foundStep) return foundStep.id;
    
    // Fallback based on progress percentage
    if (progress < 20) return 'upload';
    if (progress < 40) return 'motion';  
    if (progress < 60) return 'depth';
    if (progress < 80) return 'render';
    return 'finalize';
  };

  const currentStep = getCurrentStep();
  const currentStepIndex = steps.findIndex(step => step.id === currentStep);

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-foreground text-center">
        Processing Pipeline
      </h3>
      
      <div className="space-y-4">
        {steps.map((step, index) => {
          const isCompleted = index < currentStepIndex || progress >= 95;
          const isActive = index === currentStepIndex && progress < 95;
          const isPending = index > currentStepIndex && progress < 95;
          
          const Icon = step.icon;
          
          return (
            <motion.div
              key={step.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
              className={`
                flex items-center gap-4 p-3 rounded-lg border transition-all duration-500
                ${isCompleted ? 'border-vr-success bg-vr-success/10' : ''}
                ${isActive ? 'border-vr-primary bg-vr-primary/10' : ''}
                ${isPending ? 'border-border bg-card/50' : ''}
              `}
            >
              {/* Step icon */}
              <div className={`
                w-10 h-10 rounded-full flex items-center justify-center transition-all duration-500
                ${isCompleted ? 'bg-vr-success' : ''}
                ${isActive ? 'bg-vr-primary animate-pulse-glow' : ''}
                ${isPending ? 'bg-muted' : ''}
              `}>
                {isCompleted ? (
                  <CheckCircle2 className="w-5 h-5 text-white" />
                ) : isActive ? (
                  <Loader2 className="w-5 h-5 text-white animate-spin" />
                ) : (
                  <Icon className={`w-5 h-5 ${isPending ? 'text-muted-foreground' : 'text-white'}`} />
                )}
              </div>
              
              {/* Step content */}
              <div className="flex-1">
                <p className={`
                  font-medium text-sm transition-colors duration-500
                  ${isCompleted ? 'text-vr-success' : ''}
                  ${isActive ? 'text-vr-primary' : ''}
                  ${isPending ? 'text-muted-foreground' : ''}
                `}>
                  {step.label}
                </p>
                
                {isActive && currentMessage && (
                  <motion.p
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-xs text-muted-foreground mt-1"
                  >
                    {currentMessage}
                  </motion.p>
                )}
              </div>
              
              {/* Connection line */}
              {index < steps.length - 1 && (
                <div className={`
                  absolute left-8 top-16 w-0.5 h-6 transition-colors duration-500
                  ${isCompleted ? 'bg-vr-success' : 'bg-border'}
                `} style={{ marginTop: '0.75rem' }} />
              )}
            </motion.div>
          );
        })}
      </div>
      
      {/* Progress indicator */}
      <div className="mt-6">
        <div className="flex items-center justify-between text-sm mb-2">
          <span className="text-muted-foreground">Overall Progress</span>
          <span className="text-vr-primary font-medium">{Math.round(progress)}%</span>
        </div>
        <div className="w-full bg-muted rounded-full h-2">
          <motion.div
            className="bg-gradient-to-r from-vr-primary to-vr-secondary h-2 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.5, ease: "easeOut" }}
          />
        </div>
      </div>
    </div>
  );
};

export default ProcessingTimeline;