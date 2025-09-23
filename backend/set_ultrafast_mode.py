#!/usr/bin/env python3
"""
Ultra-Fast Mode Toggle Script

This script enables or disables ultra-fast processing mode for faster VR180 generation.
Ultra-fast mode trades some quality for significantly faster processing times.
"""

import sys
import os
from pathlib import Path

def toggle_ultrafast_mode(enable=True):
    """
    Toggle ultra-fast mode in config.py
    """
    config_path = Path(__file__).parent / "config.py"
    
    if not config_path.exists():
        print(f"Error: config.py not found at {config_path}")
        return False
    
    # Read current config
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Find and replace the ULTRAFAST_MODE line
    lines = content.split('\n')
    new_lines = []
    
    found_ultrafast = False
    for line in lines:
        if line.strip().startswith('ULTRAFAST_MODE'):
            new_lines.append(f'ULTRAFAST_MODE = {str(enable).lower()}  # Enable for fastest possible processing (lower quality)')
            found_ultrafast = True
        else:
            new_lines.append(line)
    
    # If we didn't find the line, add it
    if not found_ultrafast:
        # Find the line before "# Misc"
        for i, line in enumerate(new_lines):
            if line.strip() == '# Misc':
                new_lines.insert(i, f'ULTRAFAST_MODE = {str(enable).lower()}  # Enable for fastest possible processing (lower quality)')
                new_lines.insert(i, '# Ultra-fast processing mode')
                break
    
    # Write updated config
    with open(config_path, 'w') as f:
        f.write('\n'.join(new_lines))
    
    print(f"Ultra-fast mode {'enabled' if enable else 'disabled'} successfully!")
    print(f"  - Input target height: {'240px' if enable else '360px'}")
    print(f"  - Key frame interval: {'5 frames' if enable else '10 frames'}")
    print(f"  - Processing speed: {'~8x faster' if enable else 'normal'}")
    print(f"  - Output quality: {'much lower' if enable else 'normal'}")
    
    return True

def show_status():
    """
    Show current ultra-fast mode status
    """
    config_path = Path(__file__).parent / "config.py"
    
    if not config_path.exists():
        print(f"Error: config.py not found at {config_path}")
        return
    
    # Read current config
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Check if ultra-fast mode is enabled
    if 'ULTRAFAST_MODE = true' in content.lower():
        print("Ultra-fast mode: ENABLED")
        print("  - Processing speed: ~8x faster")
        print("  - Output quality: much lower")
        print("  - Input target height: 240px")
        print("  - Key frame interval: 5 frames")
    else:
        print("Ultra-fast mode: DISABLED")
        print("  - Processing speed: normal")
        print("  - Output quality: normal")
        print("  - Input target height: 360px")
        print("  - Key frame interval: 10 frames")

# Enable ultra-fast mode by default for minimal processing
if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Enable ultra-fast mode by default for minimal processing
        toggle_ultrafast_mode(True)
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    if command in ['on', 'enable', 'true', '1']:
        toggle_ultrafast_mode(True)
    elif command in ['off', 'disable', 'false', '0']:
        toggle_ultrafast_mode(False)
    else:
        print("Invalid command. Use 'on' or 'off'")
        sys.exit(1)