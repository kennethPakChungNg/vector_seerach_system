"""
Embedding tracker module for managing incremental updates to product embeddings.
"""

import os
import json
import time
import pandas as pd
import numpy as np
import faiss
from typing import Dict, List, Tuple, Optional

class EmbeddingTracker:
    """
    Track and manage product embeddings for incremental updates.
    """
    
    def __init__(self, base_dir: str, metadata_file: str = "embedding_metadata.json"):
        """
        Initialize the embedding tracker.
        
        Args:
            base_dir: Base directory for storing metadata and embeddings
            metadata_file: Name of the metadata file
        """
        self.base_dir = base_dir
        self.metadata_path = os.path.join(base_dir, metadata_file)
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load metadata from file or create default metadata."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading metadata file: {e}")
                return self._create_default_metadata()
        else:
            return self._create_default_metadata()
    
    def _create_default_metadata(self) -> Dict:
        """Create default metadata structure."""
        return {
            "last_update_time": 0,
            "product_map": {},  # Maps product_id to index position
            "total_products": 0,
            "embedding_dimension": 0,
            "updates_history": []
        }
    
    def _save_metadata(self):
        """Save metadata to file."""
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_products_to_update(self, df: pd.DataFrame, 
                              modified_time_col: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Identify products that need to be added or updated.
        
        Args:
            df: DataFrame containing all products
            modified_time_col: Column containing modification timestamps (optional)
            
        Returns:
            Tuple of (new_products, updated_products) DataFrames
        """
        all_product_ids = set(df['product_id'].values)
        existing_product_ids = set(self.metadata["product_map"].keys())
        
        # Find new products
        new_product_ids = all_product_ids - existing_product_ids
        new_products = df[df['product_id'].isin(new_product_ids)].copy()
        
        # Find updated products (if timestamp column is provided)
        updated_products = pd.DataFrame()
        if modified_time_col and modified_time_col in df.columns:
            last_update = self.metadata["last_update_time"]
            potential_updates = df[
                (df['product_id'].isin(existing_product_ids)) & 
                (df[modified_time_col] > last_update)
            ].copy()
            updated_products = potential_updates
        
        return new_products, updated_products
    
    def update_embeddings(self, 
                         df: pd.DataFrame, 
                         embeddings_generator,
                         index_path: str,
                         text_column: str = 'combined_text_improved',
                         modified_time_col: str = None,
                         batch_size: int = 16) -> Tuple[np.ndarray, faiss.Index]:
        """
        Update embeddings for new or modified products.
        
        Args:
            df: DataFrame containing all products
            embeddings_generator: Function to generate embeddings
            index_path: Path to FAISS index
            text_column: Column containing text to embed
            modified_time_col: Column containing modification timestamps
            batch_size: Batch size for embedding generation
            
        Returns:
            Tuple of (all_embeddings, updated_index)
        """
        # Identify products to update
        new_products, updated_products = self.get_products_to_update(df, modified_time_col)
        products_to_update = pd.concat([new_products, updated_products])
        
        if len(products_to_update) == 0:
            print("No products need updating.")
            # Load existing embeddings and index
            all_embeddings = np.load(os.path.join(self.base_dir, "combined_embeddings.npy"))
            index = faiss.read_index(index_path)
            return all_embeddings, index
        
        print(f"Updating embeddings for {len(products_to_update)} products:")
        print(f"- {len(new_products)} new products")
        print(f"- {len(updated_products)} modified products")
        
        # Generate embeddings for products to update
        update_embeddings = embeddings_generator.generate_product_embeddings(
            df=products_to_update,
            text_column=text_column,
            batch_size=batch_size
        )
        
        # Load existing embeddings and index if they exist
        if os.path.exists(os.path.join(self.base_dir, "combined_embeddings.npy")):
            all_embeddings = np.load(os.path.join(self.base_dir, "combined_embeddings.npy"))
            index = faiss.read_index(index_path)
            
            # Set embedding dimension in metadata if not already set
            if self.metadata["embedding_dimension"] == 0:
                self.metadata["embedding_dimension"] = all_embeddings.shape[1]
        else:
            # First time setup
            all_embeddings = update_embeddings
            
            # Create new index
            dimension = update_embeddings.shape[1]
            self.metadata["embedding_dimension"] = dimension
            
            index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(update_embeddings)
            index.add(update_embeddings)
            
            # Update metadata for all new products
            for i, product_id in enumerate(new_products['product_id'].values):
                self.metadata["product_map"][product_id] = i
            
            self.metadata["total_products"] = len(new_products)
            self.metadata["last_update_time"] = time.time()
            self._save_metadata()
            
            return all_embeddings, index
        
        # Handle the update of existing products
        if len(updated_products) > 0:
            for idx, row in updated_products.iterrows():
                product_id = row['product_id']
                if product_id in self.metadata["product_map"]:
                    index_pos = self.metadata["product_map"][product_id]
                    # Remove the old vector from the index (not directly supported by FAISS)
                    # Instead, we'll rebuild the index later
                    
                    # Update the embedding in the array
                    update_pos = products_to_update[products_to_update['product_id'] == product_id].index[0]
                    all_embeddings[index_pos] = update_embeddings[update_pos - len(new_products)]
        
        # Handle new products
        if len(new_products) > 0:
            # Append new embeddings
            start_idx = len(all_embeddings)
            new_product_embeddings = update_embeddings[:len(new_products)]
            all_embeddings = np.vstack([all_embeddings, new_product_embeddings])
            
            # Update metadata
            for i, product_id in enumerate(new_products['product_id'].values):
                self.metadata["product_map"][product_id] = start_idx + i
        
        # Rebuild the index from all embeddings
        dimension = all_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(all_embeddings)
        index.add(all_embeddings)
        
        # Update metadata
        self.metadata["total_products"] = len(all_embeddings)
        self.metadata["last_update_time"] = time.time()
        self.metadata["updates_history"].append({
            "timestamp": time.time(),
            "new_products": len(new_products),
            "updated_products": len(updated_products)
        })
        self._save_metadata()
        
        # Save updated embeddings
        np.save(os.path.join(self.base_dir, "combined_embeddings.npy"), all_embeddings)
        faiss.write_index(index, index_path)
        
        return all_embeddings, index