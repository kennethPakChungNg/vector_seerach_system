# vectorshop/vectorshop/data/language/models/deepseek.py

import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ..base import BaseDetector, LanguageDetectionResult

class DeepSeekDetector(BaseDetector):
    """Language detection using specialized language model."""

    def __init__(self, config=None):
        """
        Initialize with the xlm-roberta model for better language detection.
        Args:
            config (dict): Detector-specific config, e.g. { 'max_length': 128, 'device': 'cuda', ... }
        """
        self.config = config or {}
        self.model_name = "papluca/xlm-roberta-base-language-detection"
        self.labels = {
            0: 'pt',
            1: 'en',
            2: 'unknown'
        }
        self._initialize_model()

    def _initialize_model(self):
        """Initialize model with proper error handling."""
        import torch
        
        # If 'device' is in config, use it; otherwise default to GPU if available
        self.device = self.config.get('device', "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            print(f"Successfully loaded {self.model_name}")
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise
    
    def detect(self, text: str) -> LanguageDetectionResult:
        import torch
        start_time = time.time()
        try:
            # Preprocess and tokenize text
            max_len = self.config.get('max_length', 128)
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                confidence, prediction = torch.max(probs, dim=-1)
                
                # Safely convert tensor values
                if hasattr(confidence, 'item'):
                    conf_value = confidence.item()
                else:
                    conf_value = float(confidence)
                if hasattr(prediction, 'item'):
                    pred_value = prediction.item()
                else:
                    pred_value = int(prediction)
                
                # Map to language
                language = 'pt' if pred_value == 6 else 'en' if pred_value == 13 else 'unknown'
                
                return LanguageDetectionResult(
                    text=text,
                    language=language,
                    confidence=conf_value,
                    method_used="deepseek",
                    processing_time=time.time() - start_time
                )
        
        except Exception as e:
            print(f"Error in DeepSeek detection: {str(e)}")
            return LanguageDetectionResult(
                text=text,
                language="unknown",
                confidence=0.0,
                method_used="deepseek_failed",
                processing_time=time.time() - start_time
            )


    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'model') and self.device == 'cuda':
            try:
                self.model.cpu()
                del self.model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error cleaning up: {e}")