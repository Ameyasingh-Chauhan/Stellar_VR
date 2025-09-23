import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { X, Bell, Mail, Smartphone, CheckCircle2, AlertCircle, Zap } from 'lucide-react';

interface NotificationsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const NotificationsModal: React.FC<NotificationsModalProps> = ({ isOpen, onClose }) => {
  const [settings, setSettings] = useState({
    processingComplete: true,
    processingStarted: false,
    emailNotifications: true,
    browserNotifications: true,
    errorAlerts: true,
    weeklyUpdates: false,
    marketingEmails: false
  });

  if (!isOpen) return null;

  const toggleSetting = (key: keyof typeof settings) => {
    setSettings(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const notificationTypes = [
    {
      key: 'processingComplete' as const,
      icon: CheckCircle2,
      title: 'Processing Complete',
      description: 'Get notified when your video conversion is finished',
      color: 'text-green-500'
    },
    {
      key: 'processingStarted' as const,
      icon: Zap,
      title: 'Processing Started',
      description: 'Confirmation when your video processing begins',
      color: 'text-blue-500'
    },
    {
      key: 'errorAlerts' as const,
      icon: AlertCircle,
      title: 'Error Alerts',
      description: 'Immediate alerts for processing errors or issues',
      color: 'text-red-500'
    }
  ];

  const communicationPrefs = [
    {
      key: 'emailNotifications' as const,
      icon: Mail,
      title: 'Email Notifications',
      description: 'Receive notifications via email'
    },
    {
      key: 'browserNotifications' as const,
      icon: Smartphone,
      title: 'Browser Notifications',
      description: 'Show desktop notifications in your browser'
    },
    {
      key: 'weeklyUpdates' as const,
      icon: Bell,
      title: 'Weekly Updates',
      description: 'Get weekly summaries of your VR conversions'
    }
  ];

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
              <Bell className="w-5 h-5 text-white" />
            </div>
            <h2 className="text-xl font-bold text-foreground">Notification Settings</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-muted transition-colors"
          >
            <X className="w-5 h-5 text-muted-foreground" />
          </button>
        </div>

        <div className="space-y-6">
          {/* Processing Notifications */}
          <div>
            <h3 className="font-semibold text-foreground mb-4">Processing Notifications</h3>
            <div className="space-y-3">
              {notificationTypes.map((type) => {
                const Icon = type.icon;
                return (
                  <div key={type.key} className="flex items-center justify-between p-3 rounded-xl bg-muted/30">
                    <div className="flex items-start gap-3">
                      <Icon className={`w-5 h-5 ${type.color} mt-0.5`} />
                      <div>
                        <p className="font-medium text-foreground text-sm">{type.title}</p>
                        <p className="text-xs text-muted-foreground">{type.description}</p>
                      </div>
                    </div>
                    <button
                      onClick={() => toggleSetting(type.key)}
                      className={`w-10 h-6 rounded-full transition-colors relative ${
                        settings[type.key] ? 'bg-vr-primary' : 'bg-muted'
                      }`}
                    >
                      <div
                        className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                          settings[type.key] ? 'translate-x-5' : 'translate-x-1'
                        }`}
                      />
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Communication Preferences */}
          <div>
            <h3 className="font-semibold text-foreground mb-4">Communication Preferences</h3>
            <div className="space-y-3">
              {communicationPrefs.map((pref) => {
                const Icon = pref.icon;
                return (
                  <div key={pref.key} className="flex items-center justify-between p-3 rounded-xl bg-muted/30">
                    <div className="flex items-start gap-3">
                      <Icon className="w-5 h-5 text-vr-primary mt-0.5" />
                      <div>
                        <p className="font-medium text-foreground text-sm">{pref.title}</p>
                        <p className="text-xs text-muted-foreground">{pref.description}</p>
                      </div>
                    </div>
                    <button
                      onClick={() => toggleSetting(pref.key)}
                      className={`w-10 h-6 rounded-full transition-colors relative ${
                        settings[pref.key] ? 'bg-vr-primary' : 'bg-muted'
                      }`}
                    >
                      <div
                        className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                          settings[pref.key] ? 'translate-x-5' : 'translate-x-1'
                        }`}
                      />
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Save Button */}
          <button
            onClick={() => {
              console.log('Notification settings saved:', settings);
              onClose();
            }}
            className="w-full py-3 px-4 bg-gradient-to-r from-vr-primary to-vr-secondary text-white font-medium rounded-xl hover:shadow-lg transition-all duration-200"
          >
            Save Settings
          </button>
        </div>
      </motion.div>
    </div>
  );
};