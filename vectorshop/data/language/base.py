# vectorshop/vectorshop/data/language/base.py

from dataclasses import dataclass
import time
from typing import Optional

@dataclass
class LanguageDetectionResult:
    """Common result format for all language detectors."""
    text: str
    language: str
    confidence: float
    method_used: str
    processing_time: float

class BaseDetector:
    """Base class for all language detectors."""
    
    def detect(self, text: str) -> LanguageDetectionResult:
        """Base detection method to be implemented by subclasses."""
        raise NotImplementedError
        
    def _time_execution(self, start_time: float) -> float:
        """Calculate execution time."""
        return time.time() - start_time