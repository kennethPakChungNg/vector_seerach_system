import logging
from pathlib import Path
import time
from typing import Dict, Any

class LanguageDetectionLogger:
    """Logger for language detection operations."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger('language_detection')
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = log_dir / f"detection_{time.strftime('%Y%m%d_%H%M%S')}.log"
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(handler)
    
    def log_detection(self, text: str, result: Dict[str, Any], detector: str):
        """Log a single detection result."""
        self.logger.info(
            f"Detector: {detector}\n"
            f"Text: {text[:100]}...\n"  # First 100 chars
            f"Language: {result['language']}\n"
            f"Confidence: {result['confidence']:.2f}\n"
            f"Processing Time: {result['processing_time']*1000:.2f}ms\n"
        )
    
    def log_error(self, text: str, error: Exception, detector: str):
        """Log detection errors."""
        self.logger.error(
            f"Error in {detector} detector:\n"
            f"Text: {text[:100]}...\n"
            f"Error: {str(error)}\n"
        )