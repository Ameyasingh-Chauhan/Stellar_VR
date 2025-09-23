import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { X, Settings, Zap, Download, Palette, Gauge, Video } from 'lucide-react';

interface PreferencesModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentTheme: 'dark' | 'light';
  onThemeChange: () => void;
}

export const PreferencesModal: React.FC<PreferencesModalProps> = ({ 
  isOpen, 
  onClose, 
  currentTheme, 
  onThemeChange 
}) => {
  const [preferences, setPreferences] = useState({
    defaultQuality: 'high' as 'high' | 'medium' | 'low',
    defaultLayout: 'side-by-side' as 'side-by-side' | 'over-under',
    autoDownload: false,
    useNewPipeline: true,
    showProcessingDetails: true,
    enableAnimations: true,
    autoPlay: true
  });

  if (!isOpen) return null;

  const togglePreference = (key: keyof typeof preferences) => {
    if (typeof preferences[key] === 'boolean') {
      setPreferences(prev => ({ ...prev, [key]: !prev[key] }));
    }
  };

  const updatePreference = (key: keyof typeof preferences, value: any) => {
    setPreferences(prev => ({ ...prev, [key]: value }));
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 20 }}
        className="bg-card border border-border rounded-2xl p-6 max-w-md w-full shadow-2xl max-h-[80vh] overflow-y-auto"
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-r from-vr-primary to-vr-secondary rounded-lg flex items-center justify-center">
              <Settings className="w-5 h-5 text-white" />
            </div>
            <h2 className="text-xl font-bold text-foreground">Preferences</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-muted transition-colors"
          >
            <X className="w-5 h-5 text-muted-foreground" />
          </button>
        </div>

        <div className="space-y-6">
          {/* Conversion Defaults */}
          <div>
            <h3 className="font-semibold text-foreground mb-4 flex items-center gap-2">
              <Video className="w-4 h-4 text-vr-primary" />
              Conversion Defaults
            </h3>
            <div className="space-y-4">
              {/* Default Quality */}
              <div>
                <label className="text-sm font-medium text-foreground mb-2 block">Default Quality</label>
                <select
                  value={preferences.defaultQuality}
                  onChange={(e) => updatePreference('defaultQuality', e.target.value)}
                  className="w-full p-2 bg-muted border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-vr-primary/50"
                >
                  <option value="high">High Quality</option>
                  <option value="medium">Medium Quality</option>
                  <option value="low">Low Quality</option>
                </select>
              </div>

              {/* Default Layout */}
              <div>
                <label className="text-sm font-medium text-foreground mb-2 block">Default Layout</label>
                <select
                  value={preferences.defaultLayout}
                  onChange={(e) => updatePreference('defaultLayout', e.target.value)}
                  className="w-full p-2 bg-muted border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-vr-primary/50"
                >
                  <option value="side-by-side">Side by Side</option>
                  <option value="over-under">Top Bottom</option>
                </select>
              </div>

              {/* Pipeline Toggle */}
              <div className="flex items-center justify-between p-3 rounded-xl bg-muted/30">
                <div className="flex items-start gap-3">
                  <Zap className="w-5 h-5 text-vr-primary mt-0.5" />
                  <div>
                    <p className="font-medium text-foreground text-sm">Always Use Round-2 Pipeline</p>
                    <p className="text-xs text-muted-foreground">Enhanced VR processing with cinematic effects</p>
                  </div>
                </div>
                <button
                  onClick={() => togglePreference('useNewPipeline')}
                  className={`w-10 h-6 rounded-full transition-colors relative ${
                    preferences.useNewPipeline ? 'bg-vr-primary' : 'bg-muted'
                  }`}
                >
                  <div
                    className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                      preferences.useNewPipeline ? 'translate-x-5' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
            </div>
          </div>

          {/* Behavior Settings */}
          <div>
            <h3 className="font-semibold text-foreground mb-4 flex items-center gap-2">
              <Gauge className="w-4 h-4 text-vr-primary" />
              Behavior
            </h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 rounded-xl bg-muted/30">
                <div className="flex items-start gap-3">
                  <Download className="w-5 h-5 text-vr-primary mt-0.5" />
                  <div>
                    <p className="font-medium text-foreground text-sm">Auto Download</p>
                    <p className="text-xs text-muted-foreground">Automatically download when processing completes</p>
                  </div>
                </div>
                <button
                  onClick={() => togglePreference('autoDownload')}
                  className={`w-10 h-6 rounded-full transition-colors relative ${
                    preferences.autoDownload ? 'bg-vr-primary' : 'bg-muted'
                  }`}
                >
                  <div
                    className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                      preferences.autoDownload ? 'translate-x-5' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>

              <div className="flex items-center justify-between p-3 rounded-xl bg-muted/30">
                <div className="flex items-start gap-3">
                  <Video className="w-5 h-5 text-vr-primary mt-0.5" />
                  <div>
                    <p className="font-medium text-foreground text-sm">Auto Play Videos</p>
                    <p className="text-xs text-muted-foreground">Start playing converted videos automatically</p>
                  </div>
                </div>
                <button
                  onClick={() => togglePreference('autoPlay')}
                  className={`w-10 h-6 rounded-full transition-colors relative ${
                    preferences.autoPlay ? 'bg-vr-primary' : 'bg-muted'
                  }`}
                >
                  <div
                    className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                      preferences.autoPlay ? 'translate-x-5' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
            </div>
          </div>

          {/* Appearance */}
          <div>
            <h3 className="font-semibold text-foreground mb-4 flex items-center gap-2">
              <Palette className="w-4 h-4 text-vr-primary" />
              Appearance
            </h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 rounded-xl bg-muted/30">
                <div>
                  <p className="font-medium text-foreground text-sm">Theme</p>
                  <p className="text-xs text-muted-foreground">Choose your preferred theme</p>
                </div>
                <button
                  onClick={onThemeChange}
                  className="px-3 py-1 bg-vr-primary/10 text-vr-primary rounded-lg text-sm hover:bg-vr-primary/20 transition-colors"
                >
                  {currentTheme === 'dark' ? 'Dark' : 'Light'}
                </button>
              </div>

              <div className="flex items-center justify-between p-3 rounded-xl bg-muted/30">
                <div>
                  <p className="font-medium text-foreground text-sm">Animations</p>
                  <p className="text-xs text-muted-foreground">Enable smooth animations and transitions</p>
                </div>
                <button
                  onClick={() => togglePreference('enableAnimations')}
                  className={`w-10 h-6 rounded-full transition-colors relative ${
                    preferences.enableAnimations ? 'bg-vr-primary' : 'bg-muted'
                  }`}
                >
                  <div
                    className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                      preferences.enableAnimations ? 'translate-x-5' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
            </div>
          </div>

          {/* Save Button */}
          <button
            onClick={() => {
              console.log('Preferences saved:', preferences);
              onClose();
            }}
            className="w-full py-3 px-4 bg-gradient-to-r from-vr-primary to-vr-secondary text-white font-medium rounded-xl hover:shadow-lg transition-all duration-200"
          >
            Save Preferences
          </button>
        </div>
      </motion.div>
    </div>
  );
};