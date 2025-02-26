# Add re_rank_results to the exposed functions:
from .text_processing import clean_text
from .performance import measure_performance
from .statistics import get_language_statistics
from .logging import LanguageDetectionLogger
from .visualization import LanguageDetectionVisualizer
from .deepseek_rerank import re_rank_results
from .ensemble import ensemble_detector_from_results, try_detectors_ensemble
from .deepseek_enhancer import *

__all__ = [
    'clean_text',
    'measure_performance',
    'get_language_statistics',
    'LanguageDetectionLogger',
    'LanguageDetectionVisualizer',
    're_rank_results',
    'ensemble_detector_from_results',
    'try_detectors_ensemble',
    'get_deepseek_model',
    'load_model',
    'analyze_query',
    '_extract_query_info',
    'rerank_results',
    'analyze_with_prompt'
    
]
