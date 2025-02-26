import pytest
from typing import List, Dict
from ..models import WordListDetector, DeepSeekDetector, GoogleLanguageDetector

class TestLanguageDetectors:
    """Test suite for language detectors."""
    
    @pytest.fixture
    def test_cases(self) -> List[Dict]:
        """Test cases with known languages."""
        return [
            {
                'text': "Hello, this is an English text",
                'expected_language': 'en',
                'min_confidence': 0.8
            },
            {
                'text': "Olá, este é um texto em português",
                'expected_language': 'pt',
                'min_confidence': 0.8
            },
            {
                'text': "Product muito bom, great service",
                'expected_language': 'unknown',  # Mixed language
                'min_confidence': 0.5
            }
        ]
    
    def test_word_list_detector(self, test_cases):
        detector = WordListDetector()
        for case in test_cases:
            result = detector.detect(case['text'])
            if case['expected_language'] != 'unknown':
                assert result.language == case['expected_language']
                assert result.confidence >= case['min_confidence']
    
    def test_deepseek_detector(self, test_cases):
        detector = DeepSeekDetector()
        for case in test_cases:
            result = detector.detect(case['text'])
            if case['expected_language'] != 'unknown':
                assert result.language == case['expected_language']
                assert result.confidence >= case['min_confidence']