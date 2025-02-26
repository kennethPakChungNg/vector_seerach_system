from typing import Dict, Optional, Set
import re
from dataclasses import dataclass

@dataclass
class LanguageDetectionResult:
    """Contains the results of language detection with confidence score."""
    text: str
    language: str
    confidence: float
    method_used: str  # 'word_list' or 'deep_learning'
    processing_time: float  # in seconds

class WordListDetector:
    """Fast language detection using curated word lists."""
    
    def __init__(self):
        # Portuguese word sets with confidence weights
        self.portuguese_indicators: Dict[str, float] = {
            # High confidence words (unique to Portuguese)
            'não': 0.9,
            'está': 0.9,
            'português': 0.9,
            
            # Medium confidence words (common in Portuguese reviews)
            'produto': 0.7,
            'entrega': 0.7,
            'chegou': 0.7,
            'recebi': 0.7,
            'comprei': 0.7,
            
            # Lower confidence words (might appear in other languages)
            'para': 0.5,
            'com': 0.5,
            'mas': 0.5
        }
        
        # Common Portuguese patterns
        self.pt_patterns: Set[str] = {
            r'ção\b',  # Words ending in ção
            r'ões\b',  # Plural forms ending in ões
            r'[áéíóúâêîôûãõç]',  # Portuguese-specific characters
        }
    
    def detect(self, text: str) -> LanguageDetectionResult:
        """
        Detect language using word lists and patterns.
        
        Args:
            text: Input text to analyze
            
        Returns:
            LanguageDetectionResult with language and confidence score
        """
        import time
        start_time = time.time()
        
        if not isinstance(text, str) or not text.strip():
            return LanguageDetectionResult(
                text="",
                language="unknown",
                confidence=0.0,
                method_used="word_list",
                processing_time=time.time() - start_time
            )
        
        text = text.lower().strip()
        words = set(text.split())
        
        # Calculate confidence based on word matches
        confidence = 0.0
        matched_weights = []
        
        # Check for Portuguese words
        for word in words:
            if word in self.portuguese_indicators:
                matched_weights.append(self.portuguese_indicators[word])
        
        # Check for Portuguese patterns
        pattern_matches = 0
        for pattern in self.pt_patterns:
            if re.search(pattern, text):
                pattern_matches += 1
                matched_weights.append(0.8)  # High confidence for pattern matches
        
        # Calculate final confidence
        if matched_weights:
            confidence = sum(matched_weights) / len(matched_weights)
        
        # Determine language based on confidence threshold
        language = "pt" if confidence > 0.3 else "en"
        
        return LanguageDetectionResult(
            text=text,
            language=language,
            confidence=confidence,
            method_used="word_list",
            processing_time=time.time() - start_time
        )

class SmartLanguageDetector:
    _instance = None  # Class variable to store singleton instance
    
    def __new__(cls):
        # Singleton pattern to avoid reloading model
        if cls._instance is None:
            cls._instance = super(SmartLanguageDetector, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized') or not self.initialized:
            print("Initializing language detector...")
            # Initialize word detector first (fast)
            self.word_detector = WordListDetector()
            # Lazy load deep learning model only when needed
            self.deep_detector = None
            self.confidence_threshold = 0.8
            self.initialized = True

    def detect_language(self, text: str) -> LanguageDetectionResult:
        # Try word list detection first (fast)
        quick_result = self.word_detector.detect(text)
        
        # If confident enough, return quick result
        if quick_result.confidence >= self.confidence_threshold:
            return quick_result
            
        # Only initialize deep detector if needed
        if self.deep_detector is None:
            from .deep_detector import DeepSeekDetector
            self.deep_detector = DeepSeekDetector()
            
        return self.deep_detector.detect(text)
        
        

class HybridLanguageDetector:
    def __init__(self):
        self.word_detector = WordListDetector()
        self.deepseek_detector = DeepSeekDetector()
        self.google_detector = None  # Lazy load Google detector
        self.confidence_threshold = 0.6
        
    def _init_google_detector(self):
        """Initialize Google detector only when needed."""
        if self.google_detector is None:
            try:
                from .models.google_detector import GoogleLanguageDetector
                self.google_detector = GoogleLanguageDetector()
            except Exception as e:
                print(f"Failed to initialize Google detector: {e}")
                return None
        return self.google_detector

    def detect(self, text: str) -> LanguageDetectionResult:
        # First try word list
        result = self.word_detector.detect(text)
        
        # Fallback to DeepSeek if confidence is low
        if result.confidence < self.confidence_threshold:
            deep_result = self.deepseek_detector.detect(text)
            if deep_result.confidence > result.confidence:
                result = deep_result
                
        # Final fallback to Google if still low confidence
        if result.confidence < GOOGLE_MIN_CONFIDENCE:
            google_detector = self._init_google_detector()
            if google_detector:
                try:
                    google_result = google_detector.detect(text)
                    if google_result.confidence > result.confidence:
                        result = google_result
                except Exception as e:
                    print(f"Google API fallback failed: {e}")
                
        return result
        
if __name__ == "__main__":
    from .models.deepseek import DeepSeekDetector  # Ensure DeepSeekDetector is available
    GOOGLE_MIN_CONFIDENCE = 0.5  # Define this constant if not already present

    # Test HybridLanguageDetector with mixed-language input
    detector = HybridLanguageDetector()
    result = detector.detect("bom wireless earphones")
    print("Test result for 'bom wireless earphones':")
    print(result)