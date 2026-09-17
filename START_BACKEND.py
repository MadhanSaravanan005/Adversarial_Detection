#!/usr/bin/env python3
"""
Production Backend Server
"""
import os
import sys

# Set encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run
from backend.app import app

if __name__ == '__main__':
    print("=" * 60)
    print("Automated Adversarial Monitoring & Self-Defense Backend")
    print("=" * 60)
    print("Starting server on http://127.0.0.1:9000")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    app.run(
        host='127.0.0.1',
        port=9000,
        debug=False,
        threaded=True,
        use_reloader=False
    )
