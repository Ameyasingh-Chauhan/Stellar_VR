import React from 'react';
import { QRCodeSVG } from 'qrcode.react';
import { motion } from 'framer-motion';
import { Smartphone, Copy, CheckCircle2 } from 'lucide-react';
import { useState } from 'react';

interface QRCodeDisplayProps {
  videoUrl: string;
}

export const QRCodeDisplay: React.FC<QRCodeDisplayProps> = ({ videoUrl }) => {
  const [copied, setCopied] = useState(false);

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(videoUrl);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy to clipboard:', err);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
      className="bg-card border border-border rounded-xl p-6 text-center space-y-4"
    >
      <div className="flex items-center justify-center gap-2 mb-4">
        <Smartphone className="w-5 h-5 text-vr-primary" />
        <h3 className="text-lg font-semibold text-foreground">Open in VR Headset</h3>
      </div>

      <div className="inline-block p-4 bg-white rounded-lg">
        <QRCodeSVG 
          value={videoUrl}
          size={120}
          level="M"
          includeMargin={false}
        />
      </div>

      <p className="text-sm text-muted-foreground max-w-xs mx-auto">
        Scan this QR code with your VR headset or mobile device to instantly view your VR180 video
      </p>

      <motion.button
        onClick={copyToClipboard}
        className="inline-flex items-center gap-2 px-4 py-2 bg-vr-primary/10 border border-vr-primary/20 rounded-lg text-vr-primary hover:bg-vr-primary/20 transition-colors text-sm font-medium"
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        {copied ? (
          <>
            <CheckCircle2 className="w-4 h-4" />
            Copied!
          </>
        ) : (
          <>
            <Copy className="w-4 h-4" />
            Copy Link
          </>
        )}
      </motion.button>
    </motion.div>
  );
};

export default QRCodeDisplay;