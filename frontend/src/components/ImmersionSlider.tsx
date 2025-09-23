import React from 'react';
import { motion } from 'framer-motion';
import { Zap } from 'lucide-react';

interface ImmersionSliderProps {
  value: 'low' | 'medium' | 'high';
  onChange: (value: 'low' | 'medium' | 'high') => void;
  disabled?: boolean;
}

export const ImmersionSlider: React.FC<ImmersionSliderProps> = ({ 
  value, 
  onChange, 
  disabled = false 
}) => {
  const options = [
    { 
      key: 'low' as const, 
      label: 'Subtle', 
      description: 'Gentle 3D effect',
      color: 'from-blue-500 to-cyan-500' 
    },
    { 
      key: 'medium' as const, 
      label: 'Balanced', 
      description: 'Natural immersion',
      color: 'from-vr-primary to-blue-500' 
    },
    { 
      key: 'high' as const, 
      label: 'Deep', 
      description: 'Full VR experience',
      color: 'from-vr-secondary to-vr-primary' 
    }
  ];

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <Zap className="w-4 h-4 text-vr-primary" />
        <label className="text-sm font-medium text-foreground">Immersion Level</label>
      </div>
      
      <div className="grid grid-cols-3 gap-2">
        {options.map((option) => (
          <motion.button
            key={option.key}
            onClick={() => !disabled && onChange(option.key)}
            disabled={disabled}
            className={`
              relative p-3 rounded-lg border-2 transition-all duration-300
              ${value === option.key 
                ? `border-vr-primary bg-vr-primary/10 shadow-lg shadow-vr-primary/20` 
                : 'border-border bg-card hover:border-vr-primary/50 hover:bg-vr-primary/5'
              }
              ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
            `}
            whileHover={!disabled ? { scale: 1.02 } : {}}
            whileTap={!disabled ? { scale: 0.98 } : {}}
          >
            <div className="text-center">
              <div className={`
                w-4 h-4 mx-auto mb-2 rounded-full bg-gradient-to-r ${option.color}
                ${value === option.key ? 'animate-pulse-glow' : ''}
              `} />
              <p className="font-medium text-sm text-foreground">{option.label}</p>
              <p className="text-xs text-muted-foreground">{option.description}</p>
            </div>
            
            {value === option.key && (
              <motion.div
                layoutId="immersion-indicator"
                className="absolute inset-0 border-2 border-vr-primary rounded-lg bg-vr-primary/5"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.2 }}
              />
            )}
          </motion.button>
        ))}
      </div>
      
      <div className="text-xs text-muted-foreground text-center">
        Higher levels create more pronounced depth effects
      </div>
    </div>
  );
};

export default ImmersionSlider;