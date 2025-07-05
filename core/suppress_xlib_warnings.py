#!/usr/bin/env python3
"""
Suppress Xlib.xauth warnings for headless environments
"""

import os
import sys
import warnings

def suppress_xlib_warnings():
    """Suppress Xlib warnings that occur in headless environments"""
    # Set environment variables to prevent X11 warnings
    if 'DISPLAY' not in os.environ:
        os.environ['DISPLAY'] = ':0'
    
    # Suppress Python warnings related to Xlib
    warnings.filterwarnings('ignore', category=UserWarning, module='Xlib.xauth')
    warnings.filterwarnings('ignore', message='.*Xlib.xauth.*')
    warnings.filterwarnings('ignore', message='.*xauthority.*')
    
    # Monkey patch the Xlib warning function
    try:
        import Xlib.xauth
        # Override the warning function
        Xlib.xauth.warning = lambda *args, **kwargs: None
    except ImportError:
        pass
    
    # Redirect stderr temporarily during imports that might trigger warnings
    import io
    from contextlib import redirect_stderr
    
    # Create a dummy stderr to capture warnings
    dummy_stderr = io.StringIO()
    
    return dummy_stderr

def setup_headless_environment():
    """Setup environment for headless operation"""
    # Disable any GUI backends that might cause issues
    os.environ['MPLBACKEND'] = 'Agg'  # For matplotlib
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # For Qt
    
    # Set dummy display if not set
    if 'DISPLAY' not in os.environ:
        os.environ['DISPLAY'] = ':99'
    
    # Disable audio warnings
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    
    # Suppress Xlib warnings
    suppress_xlib_warnings()