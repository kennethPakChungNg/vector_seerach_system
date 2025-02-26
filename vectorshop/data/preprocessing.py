"""
Text preprocessing module for the VectorShop project.
"""
import re
from typing import List, Optional
from unicodedata import normalize
from dataclasses import dataclass
from vectorshop.data.category_utils import get_category_info

# First, keep your existing TextPreprocessor class exactly as it is
class TextPreprocessor:
    def __init__(self, remove_accents: bool = True, 
                 remove_special_chars: bool = True,
                 min_length: int = 2):
        """
        Initialize the text preprocessor.
        
        Args:
            remove_accents: Whether to remove accents from text
            remove_special_chars: Whether to remove special characters
            min_length: Minimum length for words to keep
        """
        self.remove_accents = remove_accents
        self.remove_special_chars = remove_special_chars
        self.min_length = min_length
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by applying all preprocessing steps.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove accents if specified
        if self.remove_accents:
            text = normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
            
        # Remove special characters if specified
        if self.remove_special_chars:
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
            
        # Remove extra whitespace
        text = ' '.join(word for word in text.split() 
                       if len(word) >= self.min_length)
        
        return text
    
    def process_batch(self, texts: List[str]) -> List[str]:
        """
        Process a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of cleaned texts
        """
        return [self.clean_text(text) for text in texts]

# Keep your existing create_product_text function
def create_product_text(name: str, description: str, category: str, reviews: Optional[List[str]] = None) -> str:
    """
    Create a combined text representation of a product.
    
    Args:
        name: Product name.
        description: Product description.
        category: Product category.
        reviews: (Optional) A list of reviews.
        
    Returns:
        A string that combines the product name, category, description, and reviews.
    """
    parts = []
    if name:
        parts.append(f"Product Name: {name}")
    if category:
        # Use both primary and subcategories (first 2 levels by default)
        category_info = get_category_info(category, levels=2)
        parts.append(f"Product Hierichial Category: {category}")
    if description:
        parts.append(f"Product Description: {description}")
    if reviews:
        parts.append("Product Review Contents: " + " | ".join(reviews))
    
    return " ".join(parts)


# Now add the new ProcessedText dataclass
@dataclass
class ProcessedText:
    """Container for processed text with metadata"""
    original: str
    cleaned: str
    language: str
    has_accents: bool
    word_count: int

# And add the new MultilingualTextPreprocessor class
class MultilingualTextPreprocessor(TextPreprocessor):
    """
    Enhanced text preprocessor with multilingual support.
    Inherits from base TextPreprocessor to maintain compatibility.
    """
    def __init__(self, remove_accents: bool = False,  # Default changed to False for multilingual
                 remove_special_chars: bool = True,
                 min_length: int = 3):
        super().__init__(remove_accents, remove_special_chars, min_length)
        # Portuguese language indicators
        self.portuguese_indicators = {
            'produto', 'muito', 'bom', 'ótimo', 'excelente',
            'recomendo', 'chegou', 'recebi', 'não', 'para',
            'comprei', 'gostei', 'veio', 'até', 'mais'
        }
        
    def detect_language(self, text: str) -> str:
        """
        Detect if text is primarily Portuguese or English.
        
        Args:
            text: Input text to analyze
            
        Returns:
            'pt' for Portuguese, 'en' for English
        """
        text = text.lower()
        words = text.split()
        pt_words = sum(1 for word in words if word in self.portuguese_indicators)
        return 'pt' if pt_words / max(len(words), 1) > 0.1 else 'en'

    def process_with_metadata(self, text: str) -> ProcessedText:
        """
        Process text while preserving language information.
        
        Args:
            text: Input text to process
            
        Returns:
            ProcessedText object containing cleaned text and metadata
        """
        if not isinstance(text, str) or not text.strip():
            return ProcessedText(
                original="",
                cleaned="",
                language="unknown",
                has_accents=False,
                word_count=0
            )
        
        # Store original text
        original = text.strip()
        
        # Check for accents before cleaning
        has_accents = bool(re.search('[áéíóúãõâêîôûç]', original))
        
        # Detect language before removing accents
        language = self.detect_language(original)
        
        # Clean text using parent class method
        cleaned = self.clean_text(text)
        
        return ProcessedText(
            original=original,
            cleaned=cleaned,
            language=language,
            has_accents=has_accents,
            word_count=len(cleaned.split())
        )

    def process_product_content(self, 
                              name: str, 
                              description: str, 
                              category: str,
                              reviews: List[str]) -> ProcessedText:
        """
        Process all product text content while maintaining language information.
        
        Args:
            name: Product name
            description: Product description
            category: Product category
            reviews: List of product reviews
            
        Returns:
            ProcessedText object with combined and processed content
        """
        # Combine all text while maintaining structure
        combined = create_product_text(name, description, category)
        if reviews:
            combined += f" reviews: {' | '.join(reviews)}"
            
        # Process the combined text
        return self.process_with_metadata(combined)