import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import pandas as pd

class LanguageDetectionVisualizer:
    """Visualizes language detection results."""
    
    def plot_language_distribution(self, results: Dict[str, int], title: str = "Language Distribution"):
        """Create a pie chart of language distribution."""
        plt.figure(figsize=(10, 6))
        plt.pie(
            results.values(),
            labels=results.keys(),
            autopct='%1.1f%%',
            colors=sns.color_palette("husl", len(results))
        )
        plt.title(title)
        plt.show()
    
    def plot_confidence_distribution(self, confidence_scores_by_detector: Dict[str, List[float]]):
        """Create box plots of confidence scores by detector."""
        # Convert the dictionary into a DataFrame
        df = pd.DataFrame({
            detector: scores for detector, scores in confidence_scores_by_detector.items()
        })
        # Melt the DataFrame so each row is a single confidence score with its detector label
        df_melted = df.melt(var_name='Detector', value_name='Confidence')
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df_melted, x='Detector', y='Confidence')
        plt.title("Confidence Score Distribution by Detector")
        plt.show()

    
    def plot_processing_times(self, times: Dict[str, List[float]]):
        """Create a violin plot of processing times by detector."""
        plt.figure(figsize=(12, 6))
        data = []
        for detector, detector_times in times.items():
            data.extend([(detector, t) for t in detector_times])
        
        df = pd.DataFrame(data, columns=['Detector', 'Time (ms)'])
        sns.violinplot(data=df, x='Detector', y='Time (ms)')
        plt.title("Processing Time Distribution by Detector")
        plt.yscale('log')  # Log scale for better visualization
        plt.show()