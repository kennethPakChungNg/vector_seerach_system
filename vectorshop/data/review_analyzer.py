"""
Review analyzer module for processing product reviews.
"""

import pandas as pd
import numpy as np
import time
import os
import json
from typing import Dict, List, Tuple, Optional

class ReviewAnalyzer:
    """
    Analyze product reviews and provide sentiment scores and features.
    """
    
    def __init__(self, device="cpu", cache_dir=None, cache_ttl=86400*7):
        """
        Initialize the review analyzer.
        
        Args:
            device: Device to run models on
            cache_dir: Directory for caching review analysis
            cache_ttl: Cache time-to-live in seconds (default: 7 days)
        """
        self.device = device
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl
        self._deepseek_enhancer = None
        self.cache = self._load_cache() if cache_dir else {}
    
    def _load_cache(self) -> Dict:
        """Load review analysis cache from file."""
        cache_path = os.path.join(self.cache_dir, "review_analysis_cache.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cache = json.load(f)
                
                # Clean expired cache entries
                current_time = time.time()
                clean_cache = {}
                for key, entry in cache.items():
                    if current_time - entry.get("timestamp", 0) < self.cache_ttl:
                        clean_cache[key] = entry
                
                print(f"Loaded {len(clean_cache)} valid review analysis entries from cache.")
                return clean_cache
            except Exception as e:
                print(f"Error loading review cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save review analysis cache to file."""
        if not self.cache_dir:
            return
            
        cache_path = os.path.join(self.cache_dir, "review_analysis_cache.json")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            print(f"Error saving review cache: {e}")
    
    def _get_deepseek_enhancer(self):
        """Lazy-load the DeepSeek enhancer."""
        if self._deepseek_enhancer is None:
            from vectorshop.data.language.utils.deepseek_enhancer import DeepSeekEnhancer
            self._deepseek_enhancer = DeepSeekEnhancer(device=self.device)
        return self._deepseek_enhancer
    
    def analyze_reviews(self, product_id: str, reviews: List[str]) -> Dict:
        """
        Analyze a list of reviews for a product.
        
        Args:
            product_id: Product ID for caching
            reviews: List of review texts
            
        Returns:
            Dictionary with sentiment scores and key features
        """
        # Check cache first
        cache_key = f"{product_id}_{hash(str(reviews))}"
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            current_time = time.time()
            if current_time - cache_entry.get("timestamp", 0) < self.cache_ttl:
                return cache_entry["analysis"]
        
        # Combine reviews for analysis
        combined_reviews = "\n".join(reviews[:10])  # Limit to top 10 reviews
        
        # If no reviews, return default analysis
        if not combined_reviews.strip():
            default_analysis = {
                "sentiment_score": 5.0,  # Neutral score
                "positive_features": [],
                "negative_features": [],
                "key_concerns": [],
                "general_sentiment": "No reviews available"
            }
            return default_analysis
        
        # Analyze with DeepSeek
        enhancer = self._get_deepseek_enhancer()
        prompt = f"""
        Analyze these product reviews and extract sentiment and key feature mentions:
        
        REVIEWS:
        {combined_reviews}
        
        Return a JSON object with these fields:
        - sentiment_score: Overall sentiment score from 0 to 10 (0=very negative, 10=very positive)
        - positive_features: List of positive features mentioned in reviews
        - negative_features: List of negative features mentioned in reviews
        - key_concerns: Common problems or concerns mentioned
        - general_sentiment: Brief summary of customer sentiment
        
        Format as valid JSON only - no additional text.
        """
        
        analysis = enhancer.analyze_with_prompt(prompt)
        
        # Store in cache
        if self.cache_dir:
            self.cache[cache_key] = {
                "timestamp": time.time(),
                "analysis": analysis
            }
            self._save_cache()
        
        return analysis
    
    def get_review_score(self, product_id: str, reviews: List[str]) -> float:
        """
        Get a normalized review score (0-1) for search ranking.
        
        Args:
            product_id: Product ID
            reviews: List of review texts
            
        Returns:
            Normalized review score from 0 to 1
        """
        if not reviews:
            return 0.5  # Neutral score for products with no reviews
        
        analysis = self.analyze_reviews(product_id, reviews)
        
        # Extract sentiment score and normalize to 0-1
        sentiment_score = analysis.get("sentiment_score", 5)
        normalized_score = sentiment_score / 10.0
        
        return normalized_score