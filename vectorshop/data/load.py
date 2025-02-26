import pandas as pd
from pathlib import Path
from typing import Dict, Optional

class OlistDataLoader:
    """Handles loading and combining data from multiple Olist dataset files."""
    
    def __init__(self, data_dir: Path):
        """
        Initialize the data loader.
        
        Args:
            data_dir (Path): Directory containing the dataset files
        """
        self.data_dir = data_dir
        self.datasets: Dict[str, Optional[pd.DataFrame]] = {}
    
    def load_file(self, filename: str) -> Optional[pd.DataFrame]:
        """
        Load a single CSV file.
        
        Args:
            filename (str): Name of the file to load
            
        Returns:
            Optional[pd.DataFrame]: Loaded dataframe or None if error occurs
        """
        try:
            filepath = self.data_dir / filename
            df = pd.read_csv(filepath)
            print(f"Successfully loaded {filename} with {len(df)} rows")
            return df
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None

    def get_sample_product_data(self, n_samples: Optional[int] = None) -> pd.DataFrame:  # Changed method name to match what we're using
        """
        Get basic product data with category translations.
        
        Args:
            n_samples (Optional[int]): Number of samples to return. If None, returns all data.
            
        Returns:
            pd.DataFrame: Combined product data
        """
        # Load basic datasets
        products_df = self.load_file('olist_products_dataset.csv')
        translations_df = self.load_file('product_category_name_translation.csv')
        
        if products_df is None or translations_df is None:
            raise ValueError("Failed to load required datasets")
        
        # Merge products with translations
        combined_df = products_df.merge(
            translations_df,
            on='product_category_name',
            how='left'
        )
        
        # Sample if requested
        if n_samples is not None:
            combined_df = combined_df.sample(n=min(n_samples, len(combined_df)), random_state=42)
        
        return combined_df
    
    #  include actual product text content
    def get_full_product_data(self, n_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Get complete product data including reviews and order information.
        
        Args:
            n_samples (Optional[int]): Number of samples to return. If None, returns all data.
            
        Returns:
            pd.DataFrame: Complete product data with reviews and order information
        """
        # Load all necessary datasets
        products_df = self.load_file('olist_products_dataset.csv')
        translations_df = self.load_file('product_category_name_translation.csv')
        order_items_df = self.load_file('olist_order_items_dataset.csv')
        reviews_df = self.load_file('olist_order_reviews_dataset.csv')
        
        if any(df is None for df in [products_df, translations_df, order_items_df, reviews_df]):
            raise ValueError("Failed to load required datasets")
        
        # First, merge products with translations
        products_with_categories = products_df.merge(
            translations_df,
            on='product_category_name',
            how='left'
        )
        
        # Connect reviews to products through order_items
        reviews_with_orders = reviews_df.merge(
            order_items_df[['order_id', 'product_id']],
            on='order_id',
            how='left'
        )
        
        # Aggregate reviews by product
        product_reviews = reviews_with_orders.groupby('product_id').agg({
            'review_comment_title': lambda x: ' | '.join(filter(None, x.dropna())),
            'review_comment_message': lambda x: ' | '.join(filter(None, x.dropna())),
            'review_score': 'mean'
        }).reset_index()
        
        # Combine all information
        final_df = products_with_categories.merge(
            product_reviews,
            on='product_id',
            how='left'
        )
        
        # Sample if requested
        if n_samples is not None:
            final_df = final_df.sample(n=min(n_samples, len(final_df)), random_state=42)
        
        return final_df
    
    def load_processed_data(csv_path="/content/drive/My Drive/E-commerce_Analysis/data/processed/amazon_with_images.csv"):
        try:
            df = pd.read_csv(csv_path)
            required_cols = ['product_id', 'combined_text', 'product_link', 'image_url', 'image_path']
            if not all(col in df.columns for col in required_cols):
                raise ValueError("Missing required columns in processed data")
            return df
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
            return pd.DataFrame()