import React from 'react';
import { motion } from 'framer-motion';
import { Home, Info, Sparkles } from 'lucide-react';

interface SidebarProps {
  activeView: 'home' | 'about';
  onViewChange: (view: 'home' | 'about') => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeView, onViewChange }) => {
  const menuItems = [
    {
      id: 'home' as const,
      icon: Home,
      label: 'Conversion Hub',
      isActive: activeView === 'home'
    },
    {
      id: 'about' as const,
      icon: Info,
      label: 'About the Tech',
      isActive: activeView === 'about'
    }
  ];

  return (
    <motion.div
      initial={{ x: -100, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="w-16 bg-card border-r border-border flex flex-col items-center py-6 space-y-6"
    >
      {/* Logo */}
      <div className="w-10 h-10 bg-gradient-to-br from-vr-primary to-vr-secondary rounded-lg flex items-center justify-center animate-pulse-glow">
        <Sparkles className="w-5 h-5 text-white" />
      </div>

      {/* Navigation */}
      <nav className="flex flex-col space-y-4">
        {menuItems.map((item) => {
          const Icon = item.icon;
          
          return (
            <motion.button
              key={item.id}
              onClick={() => onViewChange(item.id)}
              className={`
                relative w-10 h-10 rounded-lg flex items-center justify-center transition-all duration-300 group
                ${item.isActive 
                  ? 'bg-vr-primary text-white shadow-lg shadow-vr-primary/25' 
                  : 'bg-transparent text-muted-foreground hover:bg-vr-primary/10 hover:text-vr-primary'
                }
              `}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
              title={item.label}
            >
              <Icon className="w-5 h-5" />
              
              {/* Active indicator - Fixed animation */}
              {item.isActive && (
                <motion.div
                  layoutId="sidebar-active-indicator"
                  className="absolute -right-1 top-1/2 w-1 h-6 bg-vr-primary rounded-l-full shadow-lg"
                  style={{ transform: 'translateY(-50%)' }}
                  initial={false}
                  animate={{ 
                    opacity: 1,
                    scale: 1
                  }}
                  transition={{ 
                    type: "spring",
                    stiffness: 500,
                    damping: 30,
                    duration: 0.3 
                  }}
                />
              )}
              
              {/* Tooltip */}
              <div className="absolute left-full ml-3 px-2 py-1 bg-foreground text-background text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-50">
                {item.label}
              </div>
            </motion.button>
          );
        })}
      </nav>
    </motion.div>
  );
};

export default Sidebar;