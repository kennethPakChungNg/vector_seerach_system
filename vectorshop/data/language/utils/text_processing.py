import re

def clean_text(text: str, preserve_accents: bool = True) -> str:
    """
    Enhanced cleaning: lowercasing, punctuation removal, and whitespace normalization.
    
    Args:
        text (str): Input text.
        preserve_accents (bool): Whether to keep accented characters.
        
    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower().strip()
    
    # Remove punctuation
    import string
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    # Optionally remove accents if not needed
    if not preserve_accents:
        import unicodedata
        text = ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text
