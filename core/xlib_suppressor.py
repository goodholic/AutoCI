#!/usr/bin/env python3
"""
More robust Xlib warning suppression
"""

import os
import sys
import io
import contextlib
import logging

class XlibSuppressor:
    """Context manager to suppress Xlib warnings"""
    
    def __init__(self):
        self.original_stderr = None
        self.devnull = None
        
    def __enter__(self):
        # Save original stderr
        self.original_stderr = sys.stderr
        
        # Redirect stderr to devnull during Xlib imports
        self.devnull = open(os.devnull, 'w')
        sys.stderr = self.devnull
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore stderr
        sys.stderr = self.original_stderr
        
        # Close devnull
        if self.devnull:
            self.devnull.close()

def suppress_all_xlib_warnings():
    """Comprehensive Xlib warning suppression"""
    # Set environment variables
    os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
    
    # Set dummy display if not set
    if 'DISPLAY' not in os.environ:
        os.environ['DISPLAY'] = ':99'
    
    # Disable X11 authentication
    os.environ['XAUTHORITY'] = '/dev/null'
    
    # Suppress logging warnings
    logging.getLogger('Xlib').setLevel(logging.ERROR)
    logging.getLogger('Xlib.xauth').setLevel(logging.ERROR)
    
    # Try to import and patch Xlib
    with XlibSuppressor():
        try:
            import Xlib.xauth
            # Override warning function
            Xlib.xauth.warning = lambda *args, **kwargs: None
        except ImportError:
            pass