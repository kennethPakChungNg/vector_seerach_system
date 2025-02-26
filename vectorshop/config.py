"""
Configuration settings for the VectorShop project.
"""
from pathlib import Path
import torch


# Project paths - Keep these unchanged
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Ensure directories exist - Keep this
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Language Detection Settings - Add this new section
LANGUAGE_DETECTION = {
    'word_list': {
        'min_confidence': 0.5,  # Threshold for word-list detector
        'boost_confidence': 0.1  # Amount to boost confidence when patterns match
    },
    'deepseek': {
        'model_name': "papluca/xlm-roberta-base-language-detection",
        'max_length': 128,
        'min_confidence': 0.7,
        'device': "cuda" if torch.cuda.is_available() else "cpu"
    },
    'google': {
        'min_confidence': 0.8,
        'credentials_path': Path("/content/drive/My Drive/E-commerce_Analysis/keys/google-service-account.json")
    }
}

GOOGLE_APPLICATION_CREDENTIALS = Path("/content/drive/My Drive/E-commerce_Analysis/keys/google-service-account.json")  

# Vector Embedding Settings - Keep this for later use
EMBEDDING_MODEL = {
    'name': "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    'max_length': 512,
    'batch_size': 32,
    'dimension': 1024
}

# Vector store settings - Keep this
INDEX_PATH = PROCESSED_DATA_DIR / "product_index.faiss"

# Data files - Keep this
DATA_FILES = {
    'products': 'olist_products_dataset.csv',
    'orders': 'olist_orders_dataset.csv',
    'order_items': 'olist_order_items_dataset.csv',
    'customers': 'olist_customers_dataset.csv',
    'translations': 'product_category_name_translation.csv'
}