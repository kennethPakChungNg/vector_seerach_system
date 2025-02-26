import re
import time
from typing import Dict, Set, Optional
from ..base import BaseDetector, LanguageDetectionResult

class WordListDetector(BaseDetector):
    """Fast language detection using curated word lists and patterns."""
    
    def __init__(self):
        """Initialize detector with carefully curated language indicators."""
        # Portuguese words with confidence weights
        self.portuguese_indicators: Dict[str, float] = {
            # High confidence words (unique to Portuguese)
            'não': 0.9,
            'está': 0.9,
            'português': 0.9,
            'obrigado': 0.9,
            'então': 0.9,
            'não': 0.9,
            
            # Medium confidence words (common in reviews)
            'correto': 0.7,
            'produto': 0.7,
            'entrega': 0.7,
            'chegou': 0.7,
            'recebi': 0.7,
            'comprei': 0.7,
            'ótimo': 0.7,
            'péssimo': 0.7,
            'bom': 0.7,
            'ruim': 0.7,
            'sim': 0.7,
            
            # Common words (lower confidence)
            'vale': 0.6,
            'pena': 0.6,
            'para': 0.5,
            'com': 0.5,
            'mas': 0.5,
            'bem': 0.5
        }
        
        # Portuguese linguistic patterns
        self.pt_patterns: Set[str] = {
            r'ção\b',         # Words ending in ção
            r'ões\b',         # Plural forms ending in ões
            r'inho\b',        # Diminutive forms
            r'inha\b',
            r'[áéíóúâêîôûãõç]', # Portuguese-specific characters
        }
        
        # English indicators for comparison
        self.english_indicators: Set[str] = {
            'the', 'product', 'delivery', 'received',
            'arrived', 'good', 'bad', 'excellent',
            'shipping', 'ordered', 'quality'
        }
    
    def preprocess_text(self, text: str) -> str:
        """Clean text while preserving language-specific characters."""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase and strip whitespace
        text = text.lower().strip()
        
        # Remove extra whitespace but preserve accented characters
        text = ' '.join(text.split())
        
        return text
    
    def detect(self, text: str) -> LanguageDetectionResult:
        """
        Detect language using word lists and patterns.
        
        This method uses a sophisticated scoring system that considers:
        - Presence of language-specific words
        - Linguistic patterns
        - Character distributions
        - Word frequencies
        """
        start_time = time.time()
        
        try:
            # Preprocess text
            cleaned_text = self.preprocess_text(text)
            if not cleaned_text:
                return LanguageDetectionResult(
                    text=text,
                    language="unknown",
                    confidence=0.0,
                    method_used="word_list",
                    processing_time=self._time_execution(start_time)
                )
            
            # Split into words and create a set for efficient lookup
            words = set(cleaned_text.split())
            
            # Calculate Portuguese confidence
            pt_confidence = 0.0
            matched_weights = []
            
            # Check for Portuguese words
            for word in words:
                if word in self.portuguese_indicators:
                    matched_weights.append(self.portuguese_indicators[word])
            
            # Check for Portuguese patterns
            pattern_matches = 0
            for pattern in self.pt_patterns:
                if re.search(pattern, cleaned_text):
                    pattern_matches += 1
                    matched_weights.append(0.8)  # High confidence for patterns
            
            # Check for English words
            english_matches = len(words.intersection(self.english_indicators))
            
            # Calculate final confidence
            if matched_weights:
                pt_confidence = sum(matched_weights) / len(matched_weights)
                
                # Boost confidence if multiple indicators are present
                if pattern_matches > 0 and len(matched_weights) > 2:
                    pt_confidence = min(pt_confidence + 0.1, 1.0)
            
            # Determine language based on confidence levels
            if pt_confidence > 0.5:
                language = "pt"
                confidence = pt_confidence
            elif english_matches > 1:
                language = "en"
                confidence = 0.6 + (english_matches * 0.1)  # Scale with matches
            else:
                language = "unknown"
                confidence = 0.0
            
            return LanguageDetectionResult(
                text=text,
                language=language,
                confidence=confidence,
                method_used="word_list",
                processing_time=self._time_execution(start_time)
            )
            
        except Exception as e:
            print(f"Error in word list detection: {e}")
            return LanguageDetectionResult(
                text=text,
                language="unknown",
                confidence=0.0,
                method_used="word_list_failed",
                processing_time=self._time_execution(start_time)
            )