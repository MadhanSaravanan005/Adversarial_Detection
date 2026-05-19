"""
Folder Monitor - Optional production component
Continuously watches for images and auto-detects them
Can run standalone but requires optional config

This module demonstrates how the detection engine can be used
in a production batch processing pipeline.
"""

import time
import logging
from pathlib import Path
from PIL import Image
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.absolute()))
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

try:
    from detection_core import detect_image_fast, get_decision
except ImportError:
    try:
        from .detection_core import detect_image_fast, get_decision
    except ImportError:
        print("ERROR: Could not import detection_core")
        sys.exit(1)

# Optional configuration - use defaults if not available
try:
    from config import MONITORED_FOLDER, FOLDER_MONITOR, LOGGING
except ImportError:
    try:
        from .config import MONITORED_FOLDER, FOLDER_MONITOR, LOGGING
    except ImportError:
        # Fallback defaults
        MONITORED_FOLDER = Path('./monitored_folders/input')
        FOLDER_MONITOR = {
            'enabled': True,
            'supported_formats': ['.jpg', '.jpeg', '.png', '.gif', '.bmp'],
            'auto_organize': True,
            'check_interval': 2
    }
    LOGGING = {'level': 'INFO'}


class BasicResultsLogger:
    """Fallback simple logger if ResultsManager not available"""
    def __init__(self):
        self.results = []

    def log_result(self, filename, decision, risk_score):
        self.results.append({
            'filename': filename,
            'decision': decision,
            'risk': risk_score,
            'timestamp': time.time()
        })
        print(f"  [Logged] {filename}: {decision} ({risk_score:.1f}%)")


class ImageEventHandler(FileSystemEventHandler):
    """Handles file system events for image processing"""

    def __init__(self, logger=None):
        self.logger = logger or BasicResultsLogger()
        self.processed_files = set()

    def on_created(self, event):
        """When a new file is created"""
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Check if it's an image
        if file_path.suffix.lower() not in FOLDER_MONITOR['supported_formats']:
            return

        # Avoid processing same file twice
        if str(file_path) in self.processed_files:
            return

        self.processed_files.add(str(file_path))

        # Small delay to ensure file is fully written
        time.sleep(0.5)

        self._process_image(file_path)

    def _process_image(self, image_path):
        """Process image for adversarial detection"""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            image = image.resize((32, 32), Image.Resampling.LANCZOS)
            image_array = np.array(image, dtype=np.float32) / 255.0

            # Detect
            risk_score, confidence, detector_scores = detect_image_fast(image_array)
            decision = get_decision(risk_score)

            # Log result
            self.logger.log_result(image_path.name, decision, risk_score)

            # Print to console
            print(f"\n{'='*60}")
            print(f"Image: {image_path.name}")
            print(f"Risk Score: {risk_score:.2f}%")
            print(f"Decision: {decision}")
            print(f"{'='*60}\n")

        except Exception as e:
            print(f"Error processing {image_path.name}: {str(e)}")


def start_folder_monitor():
    """Start monitoring folder for images"""
    if not FOLDER_MONITOR['enabled']:
        print("Folder monitor disabled in config")
        return None

    print("\n" + "="*60)
    print("FOLDER MONITOR STARTED")
    print("="*60)
    print(f"Watching: {MONITORED_FOLDER}")
    print(f"Supported formats: {', '.join(FOLDER_MONITOR['supported_formats'])}")
    print(f"Auto-organize: {FOLDER_MONITOR['auto_organize']}")
    print(f"Check interval: {FOLDER_MONITOR['check_interval']}s")
    print(f"\nDrop images into: {MONITORED_FOLDER}")
    print("="*60 + "\n")

    # Create monitored folder if it doesn't exist
    MONITORED_FOLDER.mkdir(parents=True, exist_ok=True)

    logger = BasicResultsLogger()
    event_handler = ImageEventHandler(logger)

    observer = Observer()
    observer.schedule(event_handler, str(MONITORED_FOLDER), recursive=False)
    observer.start()

    try:
        print("Monitor running. Press Ctrl+C to stop.\n")
        while True:
            time.sleep(FOLDER_MONITOR['check_interval'])

            # Display stats periodically
            if logger.results:
                total = len(logger.results)
                allowed = sum(1 for r in logger.results if r['decision'] == 'ALLOW')
                review = sum(1 for r in logger.results if r['decision'] == 'REVIEW')
                blocked = sum(1 for r in logger.results if r['decision'] == 'BLOCK')
                print(f"Stats - ALLOW: {allowed}, REVIEW: {review}, BLOCK: {blocked}, TOTAL: {total}")

    except KeyboardInterrupt:
        print("\nStopping folder monitor...")
        observer.stop()

    observer.join()


if __name__ == '__main__':
    start_folder_monitor()
