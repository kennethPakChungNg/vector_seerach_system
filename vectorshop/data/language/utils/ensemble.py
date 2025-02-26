from vectorshop.data.language.base import LanguageDetectionResult

def ensemble_detector_from_results(text: str, results: list, weights: dict) -> LanguageDetectionResult:
    """
    Combine multiple detector results using weighted confidence.
    
    Args:
        text (str): The input text.
        results (list): A list of LanguageDetectionResult objects.
        weights (dict): A mapping of detector method names to their weights.
        
    Returns:
        LanguageDetectionResult: The ensemble detection result.
    """
    weighted_scores = {}
    total_weight = 0.0

    # Combine scores
    for result in results:
        weight = weights.get(result.method_used, 1.0)
        total_weight += weight
        # Accumulate weighted score per language
        if result.language in weighted_scores:
            weighted_scores[result.language] += result.confidence * weight
        else:
            weighted_scores[result.language] = result.confidence * weight

    # Normalize scores
    for lang in weighted_scores:
        weighted_scores[lang] /= total_weight

    # Choose the language with the highest weighted confidence
    final_language = max(weighted_scores, key=weighted_scores.get)
    final_confidence = weighted_scores[final_language]

    # Use the maximum processing time from all detectors (or sum, as desired)
    max_processing_time = max(r.processing_time for r in results)
    ensemble_method = f"ensemble({','.join([r.method_used for r in results])})"

    # Create a new LanguageDetectionResult instance with the ensemble output
    return LanguageDetectionResult(
        text=text,
        language=final_language,
        confidence=final_confidence,
        method_used=ensemble_method,
        processing_time=max_processing_time
    )

def try_detectors_ensemble(text, word_list, deepseek, google, logger, config):
    """
    Run all detectors once and combine their results via a weighted ensemble.
    """
    detectors = [word_list, deepseek, google]
    # Define weights based on validation data (adjust as needed)
    weights = {
        "word_list": 1.0,
        "deepseek": 1.5,
        "google_api": 1.2,
        "google_api_failed": 0.0  # Ignored if detector fails
    }
    
    # Run each detector once and store the results
    detector_results = [detector.detect(text) for detector in detectors]
    
    # Create ensemble result from the stored results
    ensemble_result = ensemble_detector_from_results(text, detector_results, weights)
    
    # Only log if logger is not None
    if logger is not None:
        for result in detector_results:
            logger.log_detection(text, result.__dict__, result.method_used)
    
    return ensemble_result
