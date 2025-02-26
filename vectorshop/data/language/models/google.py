from google.cloud import translate_v2 as translate
from ..base import BaseDetector, LanguageDetectionResult
import time
import os
from ....config import GOOGLE_APPLICATION_CREDENTIALS

class GoogleLanguageDetector(BaseDetector):
    def __init__(self):
        """Initialize Google Cloud Translation client."""
        try:
            # Set Google credentials environment variable
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(GOOGLE_APPLICATION_CREDENTIALS)
            self.client = translate.Client()
            print("Successfully initialized Google Translation client")
        except Exception as e:
            print(f"Error initializing Google client: {e}")
            raise

    def detect(self, text: str) -> LanguageDetectionResult:
        """Detect language using Google Cloud Translation API."""
        start_time = time.time()
        
        try:
            # Call Google's language detection API
            result = self.client.detect_language(text)
            
            # Map language codes to match our format
            language = result['language']
            if language == 'pt':
                language = 'pt'
            elif language == 'en':
                language = 'en'
            else:
                language = 'unknown'
                
            return LanguageDetectionResult(
                text=text,
                language=language,
                confidence=result.get('confidence', 0.0),
                method_used="google_api",
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            print(f"Error in Google detection: {e}")
            return LanguageDetectionResult(
                text=text,
                language="unknown",
                confidence=0.0,
                method_used="google_api_failed",
                processing_time=time.time() - start_time
            )