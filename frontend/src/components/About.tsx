import React from 'react';
import { motion } from 'framer-motion';
import { 
  Eye, 
  Layers, 
  Expand, 
  Globe, 
  Focus, 
  Sparkles, 
  Package,
  Zap,
  Brain,
  Camera,
  Monitor,
  Settings,
  CheckCircle2 
} from 'lucide-react';

const ProcessStep = ({ 
  icon: Icon, 
  title, 
  description, 
  delay = 0 
}: { 
  icon: React.ComponentType<any>;
  title: string;
  description: string;
  delay?: number;
}) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.5, delay }}
    className="bg-card border border-border rounded-xl p-6 hover:shadow-lg transition-all duration-300 hover:border-vr-primary/50 group"
  >
    <div className="flex items-start gap-4">
      <div className="w-12 h-12 bg-gradient-to-br from-vr-primary to-vr-secondary rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform">
        <Icon className="w-6 h-6 text-white" />
      </div>
      <div className="flex-1">
        <h3 className="text-lg font-semibold text-foreground mb-2">{title}</h3>
        <p className="text-muted-foreground text-sm leading-relaxed">{description}</p>
      </div>
    </div>
  </motion.div>
);

export const About: React.FC = () => {
  const steps = [
    {
      icon: Eye,
      title: "Understanding Depth (MiDaS)",
      description: "We use a powerful AI model to look at every frame and create a \"depth map,\" figuring out what's near and what's far."
    },
    {
      icon: Layers,
      title: "Creating the 3D Effect (DIBR)",
      description: "Using the depth map, we generate two slightly different views—one for your left eye and one for your right—to create a true 3D effect."
    },
    {
      icon: Expand,
      title: "Expanding the View (Outpainting)",
      description: "Our AI intelligently extends the sides of your video, widening the scene to create a full 180° immersive view without black bars."
    },
    {
      icon: Globe,
      title: "Curving Into a Dome (Panoramic Projection)",
      description: "We project the expanded video onto a virtual dome, making it feel like you're sitting in the middle of the scene, just like in a real VR headset."
    },
    {
      icon: Focus,
      title: "Smoothing the Vision (Foveated Blur)",
      description: "To make the experience comfortable, we apply a subtle blur to the peripheral areas, mimicking how human eyes naturally focus and reducing eye strain."
    },
    {
      icon: Sparkles,
      title: "Making It Crisp (Super-Resolution)",
      description: "An AI upscaler enhances the video's quality, ensuring the final VR experience is sharp and clear."
    },
    {
      icon: Package,
      title: "VR-Ready Packaging (Metadata Injection)",
      description: "Finally, we package the video file with special VR180 metadata, so platforms like YouTube or VR players automatically recognize and play it in immersive 3D."
    }
  ];

  return (
    <div className="max-w-4xl mx-auto p-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-center mb-12"
      >
        <div className="inline-flex items-center gap-3 mb-6">
          <div className="w-12 h-12 bg-gradient-to-r from-vr-primary to-vr-secondary rounded-lg flex items-center justify-center animate-float">
            <Zap className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-vr-primary to-vr-secondary bg-clip-text text-transparent">
            Stellar VR180 Backend Pipeline
          </h1>
        </div>
        <h2 className="text-xl text-muted-foreground">Explained Simply</h2>
      </motion.div>

      {/* How it works section */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.2 }}
        className="mb-12"
      >
        <div className="flex items-center gap-3 mb-8">
          <Brain className="w-8 h-8 text-vr-primary" />
          <h2 className="text-2xl font-bold text-foreground">How We Transform Your Video Into VR Magic</h2>
        </div>

        <div className="grid gap-6">
          {steps.map((step, index) => (
            <ProcessStep
              key={index}
              icon={step.icon}
              title={`Step ${index + 1}: ${step.title}`}
              description={step.description}
              delay={index * 0.1}
            />
          ))}
        </div>
      </motion.div>

      {/* Technical highlights */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.8 }}
        className="bg-gradient-to-br from-vr-primary/10 to-vr-secondary/10 border border-vr-primary/20 rounded-xl p-8"
      >
        <h3 className="text-xl font-bold text-foreground mb-6 flex items-center gap-3">
          <Settings className="w-6 h-6 text-vr-primary" />
          Technical Highlights
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="flex items-start gap-3">
            <CheckCircle2 className="w-5 h-5 text-vr-success mt-1 flex-shrink-0" />
            <div>
              <p className="font-medium text-foreground">Advanced AI Processing</p>
              <p className="text-sm text-muted-foreground">Uses state-of-the-art neural networks for depth estimation and scene understanding</p>
            </div>
          </div>
          
          <div className="flex items-start gap-3">
            <CheckCircle2 className="w-5 h-5 text-vr-success mt-1 flex-shrink-0" />
            <div>
              <p className="font-medium text-foreground">Real-time Processing</p>
              <p className="text-sm text-muted-foreground">Optimized pipeline for efficient conversion without quality loss</p>
            </div>
          </div>
          
          <div className="flex items-start gap-3">
            <CheckCircle2 className="w-5 h-5 text-vr-success mt-1 flex-shrink-0" />
            <div>
              <p className="font-medium text-foreground">Universal Compatibility</p>
              <p className="text-sm text-muted-foreground">Output works with all major VR platforms and headsets</p>
            </div>
          </div>
          
          <div className="flex items-start gap-3">
            <CheckCircle2 className="w-5 h-5 text-vr-success mt-1 flex-shrink-0" />
            <div>
              <p className="font-medium text-foreground">Quality Preservation</p>
              <p className="text-sm text-muted-foreground">Maintains original video quality while adding immersive depth</p>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default About;