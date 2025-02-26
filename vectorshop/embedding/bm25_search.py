"""
BM25 search implementation for e-commerce products.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Dict, Tuple, Optional
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK resources (run once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

class BM25:
    """
    BM25 implementation for keyword search in e-commerce products.
    BM25 is a ranking function used by search engines to estimate 
    the relevance of documents to a given search query.
    """
    
    def __init__(self, k1=1.5, b=0.75):
        """
        Initialize BM25 with parameters.
        
        Args:
            k1: Term frequency saturation parameter (1.2-2.0 is reasonable)
            b: Length normalization parameter (0.75 is reasonable)
        """
        self.k1 = k1
        self.b = b
        self.vectorizer = None
        self.doc_len = None
        self.avgdl = None
        self.doc_freqs = None
        self.idf = None
        self.doc_vectors = None
        self.doc_index = None
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for BM25 indexing/searching.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Simple tokenization without using word_tokenize
        # This will split on whitespace and punctuation
        import re
        tokens = re.findall(r'\w+', text)
        
        # Remove stopwords and stem
        tokens = [self.stemmer.stem(token) for token in tokens if token.isalnum() and token not in self.stop_words]
        
        # Join back to string
        return " ".join(tokens)
    
    def fit(self, corpus: List[str]):
        """
        Fit BM25 to a corpus of documents.
        
        Args:
            corpus: List of documents
        """
        # Preprocess corpus
        processed_corpus = [self.preprocess_text(doc) for doc in corpus]
        
        # Initialize vectorizer and fit to corpus
        self.vectorizer = CountVectorizer(binary=False, min_df=2)
        self.doc_vectors = self.vectorizer.fit_transform(processed_corpus)
        
        # Calculate document lengths
        self.doc_len = np.array(self.doc_vectors.sum(axis=1)).flatten()
        self.avgdl = self.doc_len.mean()
        
        # Calculate document frequencies
        self.doc_freqs = np.array(self.doc_vectors.sum(axis=0)).flatten()
        
        # Calculate IDF scores
        n_docs = len(corpus)
        self.idf = np.log((n_docs - self.doc_freqs + 0.5) / (self.doc_freqs + 0.5) + 1.0)
        
        # Store document index for reference
        self.doc_index = list(range(len(corpus)))
        
        return self
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search the corpus for documents matching the query.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of (doc_idx, score) tuples for top_k results
        """
        if self.vectorizer is None:
            raise ValueError("BM25 must be fit to a corpus before searching")
        
        # Preprocess query
        processed_query = self.preprocess_text(query)
        
        # Vectorize query
        query_vector = self.vectorizer.transform([processed_query])
        query_terms = query_vector.indices
        
        # Calculate scores for all documents
        scores = np.zeros(len(self.doc_index))
        
        for term_idx in query_terms:
            # Get document frequencies for this term
            term_doc_freq = self.doc_vectors[:, term_idx].toarray().flatten()
            
            # Calculate BM25 score for this term across all documents
            numerator = self.idf[term_idx] * term_doc_freq * (self.k1 + 1)
            denominator = term_doc_freq + self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)
            term_scores = numerator / denominator
            
            # Add to total scores
            scores += term_scores
        
        # Get top-k document indices and scores
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]
        
        # Return (doc_idx, score) tuples
        return [(self.doc_index[idx], float(score)) for idx, score in zip(top_indices, top_scores)]

class ProductBM25Search:
    """
    BM25 search specifically for e-commerce products.
    """
    
    def __init__(self, df: pd.DataFrame, text_column='combined_text_improved'):
        """
        Initialize BM25 search for products.
        
        Args:
            df: DataFrame containing product data
            text_column: Column containing text to search
        """
        self.df = df
        self.text_column = text_column
        self.bm25 = BM25()
        self.fit()
    
    def fit(self):
        """Fit BM25 to product descriptions."""
        # Extract text
        texts = self.df[self.text_column].fillna("").astype(str).tolist()
        
        # Fit BM25
        print(f"Fitting BM25 to {len(texts)} product descriptions...")
        self.bm25.fit(texts)
        print("BM25 fitted successfully")
        
        return self
    
    def search(self, query: str, top_k: int = 20) -> pd.DataFrame:
        """
        Search for products matching the query.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            DataFrame of top matching products with scores
        """
        # Search using BM25
        results = self.bm25.search(query, top_k)
        
        # Create result DataFrame
        result_df = self.df.iloc[[idx for idx, _ in results]].copy()
        result_df['bm25_score'] = [score for _, score in results]
        
        return result_df
    
    def extract_key_terms(self, query: str, top_n: int = 5) -> List[str]:
        """
        Extract key terms from a query for feature matching.
        
        Args:
            query: Search query
            top_n: Number of top terms to extract
            
        Returns:
            List of key terms
        """
        # Preprocess query
        processed_query = self.bm25.preprocess_text(query)
        
        # Vectorize query
        query_vector = self.bm25.vectorizer.transform([processed_query])
        
        # Get term indices and their weights (IDF)
        query_terms = query_vector.indices
        term_weights = [(term, self.bm25.idf[term]) for term in query_terms]
        
        # Sort by weight and get top terms
        term_weights.sort(key=lambda x: x[1], reverse=True)
        top_terms = term_weights[:top_n]
        
        # Get term strings
        feature_names = self.bm25.vectorizer.get_feature_names_out()
        key_terms = [feature_names[term] for term, _ in top_terms]
        
        return key_terms