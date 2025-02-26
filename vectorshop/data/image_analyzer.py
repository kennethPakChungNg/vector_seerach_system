"""
Image analysis module for extracting structured features from product images.
"""

import torch
from PIL import Image
import pandas as pd
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from typing import Dict, List, Optional

class ProductImageAnalyzer:
    """Extract structured features from product images using AI models."""
    
    def __init__(self, device="cpu"):
        self.device = device
        self._blip_model = None
        self._blip_processor = None
        self._deepseek_enhancer = None
    
    def load_blip_model(self):
        """Lazily load the BLIP2 model for image captioning."""
        if self._blip_model is None:
            print("Loading BLIP2 model...")
            self._blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self._blip_model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b"
            ).to(self.device)
            print(f"BLIP2 model loaded on {self.device}")
        return self._blip_model, self._blip_processor
    
    def load_deepseek_enhancer(self):
        """Load DeepSeek model for feature extraction."""
        if self._deepseek_enhancer is None:
            # Import inside method to avoid circular imports
            from vectorshop.data.language.utils.deepseek_enhancer import DeepSeekEnhancer
            self._deepseek_enhancer = DeepSeekEnhancer(device=self.device)
        return self._deepseek_enhancer
    
    def describe_image(self, image_path: str) -> str:
        """Generate a descriptive caption for the image."""
        if pd.isna(image_path):
            return ""
            
        try:
            model, processor = self.load_blip_model()
            image = Image.open(image_path).convert("RGB")
            inputs = processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_length=50)
            
            return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception as e:
            print(f"Error describing image {image_path}: {e}")
            return ""
    
    def extract_visual_features(self, image_path: str) -> Dict:
        """
        Extract structured features from an image using BLIP2 and DeepSeek.
        
        Returns:
            Dictionary of visual features (colors, design_elements, etc.)
        """
        # First get basic image description
        image_desc = self.describe_image(image_path)
        if not image_desc:
            return {}
            
        # Use DeepSeek to extract structured features
        enhancer = self.load_deepseek_enhancer()
        
        prompt = f"""
        Extract structured product features from this image description:
        "{image_desc}"
        
        Format as a JSON object with these fields:
        - colors: List of main colors visible in the product (e.g., ["black", "silver"])
        - design_elements: Notable design characteristics (e.g., ["braided", "slim"])
        - product_type: What kind of product this appears to be
        - quality_indicators: Visual cues about product quality
        
        Format as valid JSON only - no additional text.
        """
        
        # Extract features using DeepSeek
        visual_features = enhancer.analyze_with_prompt(prompt)
        
        # Add the original description
        visual_features['original_description'] = image_desc
        
        return visual_features
    
    def batch_process_images(self, df: pd.DataFrame, image_path_col='image_path', 
                           batch_size=10, save_path=None) -> pd.DataFrame:
        """
        Process images in batches and add visual features to the dataframe.
        
        Args:
            df: DataFrame containing product data
            image_path_col: Column name for image paths
            batch_size: Number of images to process in one batch
            save_path: Optional path to save results periodically
            
        Returns:
            DataFrame with added visual_features column
        """
        result_df = df.copy()
        result_df['visual_features'] = None
        
        for i in range(0, len(df), batch_size):
            print(f"Processing batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")
            batch = df.iloc[i:i+batch_size]
            
            for idx, row in batch.iterrows():
                image_path = row[image_path_col]
                if pd.notna(image_path):
                    visual_features = self.extract_visual_features(image_path)
                    result_df.at[idx, 'visual_features'] = visual_features
            
            # Periodically save results
            if save_path and i % (batch_size * 5) == 0 and i > 0:
                result_df.to_pickle(save_path)
                print(f"Saved intermediate results to {save_path}")
                
        return result_df