from typing import List, Dict
from collections import defaultdict

def get_language_statistics(texts: List[str], detector) -> Dict:
    """Generate statistics about language distribution."""
    # Your existing get_language_statistics code with enhancements
    stats = {
        'total_texts': len(texts),
        'language_counts': defaultdict(int),
        'method_counts': defaultdict(int),
        'confidence_scores': [],
        'processing_times': [],
        'performance': {
            'avg_time': 0.0,
            'avg_confidence': 0.0
        }
    }
    
    confidences = []
    processing_times = []
    
    for text in texts:
        result = detector.detect(text)
        
        # Update language counts
        stats['language_counts'][result.language] = \
            stats['language_counts'].get(result.language, 0) + 1
            
        confidences.append(result.confidence)
        processing_times.append(result.processing_time)
    
    # Calculate averages
    if texts:
        stats['avg_confidence'] = sum(confidences) / len(texts)
        stats['avg_processing_time'] = sum(processing_times) / len(texts)
        stats['unknown_percentage'] = \
            (stats['language_counts'].get('unknown', 0) / len(texts)) * 100
    
    return stats