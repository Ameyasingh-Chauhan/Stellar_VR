#!/usr/bin/env python3
"""
Fix torchvision compatibility issues with RealESRGAN/BasicSR

This script patches the torchvision.transforms.functional_tensor import issue
that occurs with torchvision 0.23+ by adding backward compatibility.
"""

import sys
import os

def patch_torchvision_compatibility():
    """
    Patch torchvision to add backward compatibility for functional_tensor module
    """
    try:
        import torchvision
        import torchvision.transforms.functional as TF
        
        # Check if functional_tensor exists
        try:
            import torchvision.transforms.functional_tensor
        except ImportError:
            print("Patching torchvision for RealESRGAN compatibility...")
            
            # Create a mock functional_tensor module
            import types
            functional_tensor_module = types.ModuleType('torchvision.transforms.functional_tensor')
            
            # Add commonly used functions from functional
            functional_tensor_module.rgb_to_grayscale = getattr(TF, 'rgb_to_grayscale', None)
            functional_tensor_module.adjust_brightness = getattr(TF, 'adjust_brightness', None)
            functional_tensor_module.adjust_contrast = getattr(TF, 'adjust_contrast', None)
            functional_tensor_module.adjust_saturation = getattr(TF, 'adjust_saturation', None)
            functional_tensor_module.adjust_hue = getattr(TF, 'adjust_hue', None)
            
            # Add the module to sys.modules so it can be imported
            sys.modules['torchvision.transforms.functional_tensor'] = functional_tensor_module
            
            print("Patch applied successfully!")
            return True
            
    except Exception as e:
        print(f"Failed to patch torchvision compatibility: {e}")
        return False
        
    return False

if __name__ == "__main__":
    patch_torchvision_compatibility()