import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { User, LogOut, Settings, Palette, Bell, HelpCircle, ChevronDown } from 'lucide-react';

// Modals
import { HelpModal } from '@/components/HelpModal';
import { NotificationsModal } from '@/components/NotificationsModal';
import { PreferencesModal } from '@/components/PreferencesModal';

interface UserProfileProps {
  onLogout: () => void;
  theme: 'dark' | 'light';
  onThemeToggle: () => void;
}

export const UserProfile: React.FC<UserProfileProps> = ({ onLogout, theme, onThemeToggle }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [activeModal, setActiveModal] = useState<'help' | 'notifications' | 'preferences' | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const menuItems = [
    {
      icon: Palette,
      label: `Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`,
      action: () => {
        onThemeToggle();
        setIsOpen(false);
      },
      color: 'text-muted-foreground'
    },
    {
      icon: HelpCircle,
      label: 'Help & Support',
      action: () => {
        setActiveModal('help');
        setIsOpen(false);
      },
      color: 'text-muted-foreground'
    },
    {
      icon: LogOut,
      label: 'Sign Out',
      action: () => {
        onLogout();
        setIsOpen(false);
      },
      color: 'text-destructive',
      divider: true
    }
  ];

  return (
    <>
      <div className="relative" ref={dropdownRef}>
        {/* Profile Button */}
        <motion.button
          onClick={() => setIsOpen(!isOpen)}
          className="flex items-center gap-2 p-2 rounded-xl bg-muted/50 hover:bg-muted transition-all duration-200 group"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          {/* Avatar */}
          <div className="w-8 h-8 bg-gradient-to-br from-vr-primary to-vr-secondary rounded-lg flex items-center justify-center">
            <User className="w-4 h-4 text-white" />
          </div>
          
          {/* User Info */}
          <div className="text-left hidden sm:block">
            <div className="text-sm font-medium text-foreground">Admin</div>
            <div className="text-xs text-muted-foreground">Administrator</div>
          </div>
          
          {/* Dropdown Arrow */}
          <motion.div
            animate={{ rotate: isOpen ? 180 : 0 }}
            transition={{ duration: 0.2 }}
            className="text-muted-foreground group-hover:text-foreground"
          >
            <ChevronDown className="w-4 h-4" />
          </motion.div>
        </motion.button>

        {/* Dropdown Menu */}
        <AnimatePresence>
          {isOpen && (
            <motion.div
              initial={{ opacity: 0, y: -10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -10, scale: 0.95 }}
              transition={{ duration: 0.2 }}
              className="absolute right-0 top-full mt-2 w-64 bg-card border border-border rounded-xl shadow-xl z-50 overflow-hidden"
            >
              {/* User Info Header */}
              <div className="p-4 border-b border-border bg-muted/30">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-gradient-to-br from-vr-primary to-vr-secondary rounded-lg flex items-center justify-center">
                    <User className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <div className="font-medium text-foreground">Admin User</div>
                    <div className="text-sm text-muted-foreground">admin@stellarvr.com</div>
                  </div>
                </div>
              </div>

              {/* Menu Items */}
              <div className="py-2">
                {menuItems.map((item, index) => (
                  <React.Fragment key={item.label}>
                    {item.divider && <div className="h-px bg-border my-2" />}
                    <motion.button
                      onClick={() => {
                        item.action();
                      }}
                      className={`w-full flex items-center gap-3 px-4 py-3 hover:bg-muted/50 transition-colors text-left ${item.color}`}
                      whileHover={{ x: 4 }}
                      transition={{ type: "spring", stiffness: 300, damping: 25 }}
                    >
                      <item.icon className="w-4 h-4 flex-shrink-0" />
                      <span className="text-sm">{item.label}</span>
                    </motion.button>
                  </React.Fragment>
                ))}
              </div>

              {/* Footer */}
              <div className="p-3 border-t border-border bg-muted/20">
                <div className="text-xs text-muted-foreground text-center">
                  Stellar VR v1.0.0 • Logged in since today
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Modals */}
      <AnimatePresence>
        {activeModal === 'help' && (
          <HelpModal 
            isOpen={true}
            onClose={() => setActiveModal(null)}
          />
        )}
        {activeModal === 'notifications' && (
          <NotificationsModal 
            isOpen={true}
            onClose={() => setActiveModal(null)}
          />
        )}
        {activeModal === 'preferences' && (
          <PreferencesModal 
            isOpen={true}
            onClose={() => setActiveModal(null)}
            currentTheme={theme}
            onThemeChange={onThemeToggle}
          />
        )}
      </AnimatePresence>
    </>
  );
};