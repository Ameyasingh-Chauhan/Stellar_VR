import React from 'react';
import { motion } from 'framer-motion';
import { X, Mail, MessageCircle, Clock, Heart } from 'lucide-react';

interface HelpModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const HelpModal: React.FC<HelpModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  const copyEmail = () => {
    navigator.clipboard.writeText('ameyac1503@gmail.com');
    // Could add toast notification here
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 20 }}
        className="bg-card border border-border rounded-2xl p-6 max-w-md w-full shadow-2xl"
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-r from-vr-primary to-vr-secondary rounded-lg flex items-center justify-center">
              <MessageCircle className="w-5 h-5 text-white" />
            </div>
            <h2 className="text-xl font-bold text-foreground">Help & Support</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-muted transition-colors"
          >
            <X className="w-5 h-5 text-muted-foreground" />
          </button>
        </div>

        {/* Content */}
        <div className="space-y-6">
          {/* Welcome Message */}
          <div className="bg-vr-primary/5 border border-vr-primary/20 rounded-xl p-4">
            <div className="flex items-start gap-3">
              <Heart className="w-5 h-5 text-vr-primary mt-0.5 flex-shrink-0" />
              <div>
                <h3 className="font-semibold text-foreground mb-2">We're Here to Help!</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Thank you for using Stellar VR! We appreciate your patience and understanding. 
                  Our team is dedicated to providing you with the best VR conversion experience possible.
                </p>
              </div>
            </div>
          </div>

          {/* Contact Information */}
          <div className="space-y-4">
            <h3 className="font-semibold text-foreground flex items-center gap-2">
              <Mail className="w-4 h-4 text-vr-primary" />
              Get in Touch
            </h3>
            
            <div className="bg-muted/30 rounded-xl p-4 space-y-3">
              <div>
                <label className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  Support Email
                </label>
                <div className="flex items-center justify-between mt-1">
                  <span className="text-sm font-mono text-foreground">ameyac1503@gmail.com</span>
                  <button
                    onClick={copyEmail}
                    className="text-xs bg-vr-primary/10 hover:bg-vr-primary/20 text-vr-primary px-2 py-1 rounded transition-colors"
                  >
                    Copy
                  </button>
                </div>
              </div>
              
              <div className="pt-3 border-t border-border">
                <div className="flex items-start gap-2">
                  <Clock className="w-4 h-4 text-muted-foreground mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-foreground">Response Time</p>
                    <p className="text-xs text-muted-foreground">
                      We typically respond within 24-48 hours. Thank you for your patience!
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Support Guidelines */}
          <div className="space-y-3">
            <h4 className="font-medium text-foreground">When contacting support, please include:</h4>
            <ul className="text-sm text-muted-foreground space-y-1 ml-4">
              <li>• Description of the issue or question</li>
              <li>• Video file size and format (if applicable)</li>
              <li>• Processing settings used</li>
              <li>• Any error messages received</li>
            </ul>
          </div>

          {/* Close Button */}
          <button
            onClick={onClose}
            className="w-full py-3 px-4 bg-gradient-to-r from-vr-primary to-vr-secondary text-white font-medium rounded-xl hover:shadow-lg transition-all duration-200"
          >
            Got it, thanks!
          </button>
        </div>
      </motion.div>
    </div>
  );
};