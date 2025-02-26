import pandas as pd
import numpy as np
import re
import torch
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from vectorshop.data.language.utils.performance import measure_performance
from vectorshop.data.category_utils import get_primary_category
from vectorshop.data.multimodal import batch_get_image_embeddings
from vectorshop.data.language.utils.deepseek_enhancer import DeepSeekEnhancer

# Lazy-loaded model references
_text_model = None
_processor = None

def get_text_model(model_name="all-MiniLM-L6-v2", device="cpu"):
    """Lazily load the SentenceTransformer model on the specified device."""
    global _text_model
    if _text_model is None:
        _text_model = SentenceTransformer(model_name, device=device)
    return _text_model

def load_processed_data(csv_path="data/processed/processed_products.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"File {csv_path} not found!")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return pd.DataFrame()

def generate_embeddings(texts, model_name="all-MiniLM-L6-v2", device="cpu"):
    model = get_text_model(model_name, device)
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings, model

def build_faiss_index(embeddings: np.ndarray, nlist=100) -> faiss.IndexIVFFlat:
    dim = embeddings.shape[1]
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist)
    index.train(embeddings)
    index.add(embeddings)
    return index

def improved_search_multi_modal(
    query, 
    text_index, 
    image_index, 
    df, 
    model, 
    processor, 
    tfidf, 
    tfidf_matrix, 
    device="cpu", 
    top_k=5, 
    exchange_rate=83,
    use_deepseek=True
):
    """
    Improved multimodal search function with better category handling
    and optional DeepSeek reranking.
    
    Args:
        query: Search query text
        text_index: FAISS index for text embeddings
        image_index: FAISS index for image embeddings
        df: DataFrame containing product data
        model: CLIP model for embeddings
        processor: CLIP processor for preprocessing
        tfidf: TF-IDF vectorizer
        tfidf_matrix: Pre-computed TF-IDF matrix
        device: Device to use for tensor operations
        top_k: Number of results to return
        exchange_rate: Exchange rate for price conversion
        use_deepseek: Whether to use DeepSeek for query analysis and reranking
        
    Returns:
        DataFrame containing the top search results
    """
    print(f"Running improved_search_multi_modal for query: {query}")
    
    # Initialize DeepSeek enhancer if enabled
    deepseek_enhancer = None
    query_analysis = None
    if use_deepseek:
        try:
            deepseek_enhancer = DeepSeekEnhancer(device=device)
            query_analysis = deepseek_enhancer.analyze_query(query)
            print(f"Query analysis: {query_analysis}")
        except Exception as e:
            print(f"Error initializing DeepSeek: {e}")
            use_deepseek = False
    
    # Generate query embedding with CLIP
    inputs = processor(text=[query], return_tensors="pt").to(device)
    query_embedding = model.get_text_features(**inputs).detach().cpu().numpy()[0]
    query_embedding /= np.linalg.norm(query_embedding)

    # Search text index with larger candidate pool
    text_k = min(top_k * 50, len(df))
    text_scores, text_indices = text_index.search(np.expand_dims(query_embedding, 0), text_k)
    
    # Create DataFrame from text search results
    text_results = df.iloc[text_indices[0]].copy()
    text_results['text_score'] = text_scores[0]
    
    # Debug: Check for target products in initial results
    target_ids = ['B08CF3B7N1', 'B009LJ2BXA']
    for target_id in target_ids:
        if target_id in df['product_id'].values:
            target_idx = df[df['product_id'] == target_id].index[0]
            if target_idx in text_indices[0]:
                rank = list(text_indices[0]).index(target_idx) + 1
                print(f"Initial rank of {target_id}: {rank}")
            else:
                print(f"{target_id} not in initial text results")

    # Search image index with slightly smaller candidate pool
    image_k = min(top_k * 20, len(df))
    image_scores, image_indices = image_index.search(np.expand_dims(query_embedding, 0), image_k)
    
    # Create DataFrame from image search results
    image_results = df.iloc[image_indices[0]].copy()
    image_results['image_score'] = image_scores[0]

    # Combine results with deduplication
    combined_results = pd.concat([text_results, image_results]).drop_duplicates(subset='product_id')
    
    # Convert price to USD
    combined_results['price_usd'] = pd.to_numeric(
        combined_results['discounted_price'].str.replace('₹', '').str.replace(',', ''),
        errors='coerce'
    ) / exchange_rate
    combined_results = combined_results.dropna(subset=['price_usd'])

    # Extract price limit from query or query analysis
    if query_analysis and 'price_constraint' in query_analysis:
        max_price = query_analysis['price_constraint']
    else:
        price_match = re.search(r'under (\d+(\.\d+)?)\s*USD', query, re.IGNORECASE)
        max_price = float(price_match.group(1)) if price_match else None
    
    # Apply price filtering if specified
    if max_price:
        print(f"Applying price filter: < {max_price} USD")
        combined_results = combined_results[combined_results['price_usd'] < max_price]

    # Apply base scoring with higher weight for text
    combined_results['score'] = (
        combined_results['text_score'].fillna(0) * 1.5 + 
        combined_results['image_score'].fillna(0) * 0.5
    )

    # Determine relevant categories from query analysis or simple logic
    if query_analysis and 'product_type' in query_analysis:
        product_type = query_analysis['product_type'].lower()
        if product_type == 'cable':
            relevant_categories = ["USBCables", "Cable", "Charger"]
        elif product_type == 'headphone':
            relevant_categories = ["PCHeadsets", "Headphones", "Earbuds"]
        else:
            relevant_categories = [product_type]
    else:
        # Fallback to simple category detection
        query_lower = query.lower()
        if any(word in query_lower for word in ["cable", "charger", "cord"]) and "iphone" in query_lower:
            relevant_categories = ["USBCables", "Cable", "Charger"]
        elif any(word in query_lower for word in ["headset", "headphones", "earphones"]) and "noise cancelling" in query_lower:
            relevant_categories = ["PCHeadsets", "Headphones", "Earbuds"]
        else:
            relevant_categories = []
    
    print(f"Relevant categories: {relevant_categories}")

    # Debug: Check target products before boosting
    print("Before boosting:")
    print(combined_results[combined_results['product_id'].isin(target_ids)][['product_id', 'score']])

    # Apply category boost with hierarchical category handling
    category_boost = 2.0
    for index, row in combined_results.iterrows():
        # Extract all parts of hierarchical category
        category_parts = []
        if isinstance(row['category'], str):
            # Handle both | and > separators
            if '|' in row['category']:
                category_parts = [part.lower().strip() for part in row['category'].split('|')]
            elif '>' in row['category']:
                category_parts = [part.lower().strip() for part in row['category'].split('>')]
            else:
                category_parts = [row['category'].lower().strip()]
            
            # Check if any relevant category is in any part of hierarchy
            if any(rc.lower() in ' '.join(category_parts).lower() for rc in relevant_categories):
                combined_results.at[index, 'score'] += category_boost

    # Apply TF-IDF keyword boosting
    query_keywords = tfidf.transform([query])
    keyword_scores = np.dot(tfidf_matrix, query_keywords.T).toarray().ravel()

    # Apply keyword scores to results
    for idx, row in combined_results.iterrows():
        df_idx = df[df['product_id'] == row['product_id']].index[0]
        if df_idx < len(keyword_scores):
            keyword_score = keyword_scores[df_idx]
            combined_results.at[idx, 'keyword_score'] = keyword_score
            combined_results.at[idx, 'score'] += keyword_score * 1.5

    # Apply rating and popularity boosts
    if 'rating' in combined_results.columns:
        combined_results['rating'] = pd.to_numeric(combined_results['rating'], errors='coerce').fillna(0)
        combined_results['score'] += (combined_results['rating'] / 5.0) * 1.0
    
    if 'rating_count' in combined_results.columns:
        combined_results['rating_count'] = pd.to_numeric(combined_results['rating_count'], errors='coerce').fillna(0)
        max_count = combined_results['rating_count'].max()
        if max_count > 0:
            combined_results['popularity'] = combined_results['rating_count'] / max_count
            combined_results['score'] += combined_results['popularity'] * 0.5

    # Apply price normalization if max_price exists
    if max_price:
        combined_results['price_score'] = 1 - (combined_results['price_usd'] / max_price).clip(0, 1)
        combined_results['score'] += combined_results['price_score'] * 0.5

    # Debug: Check target products after boosting
    print("After boosting:")
    print(combined_results[combined_results['product_id'].isin(target_ids)][['product_id', 'score']])

    # Apply DeepSeek reranking if enabled
    if use_deepseek and deepseek_enhancer:
        try:
            # Select top candidates for reranking (more efficient)
            rerank_candidates = combined_results.sort_values('score', ascending=False).head(20)
            
            # Rerank using DeepSeek
            reranked_results = deepseek_enhancer.rerank_results(
                query=query,
                results=rerank_candidates,
                query_analysis=query_analysis
            )
            
            # Get final results
            final_results = reranked_results.head(top_k)
            print("Search results (with DeepSeek reranking):")
        except Exception as e:
            print(f"Error during DeepSeek reranking: {e}")
            # Fallback to original ranking
            final_results = combined_results.sort_values('score', ascending=False).head(top_k)
            print("Search results (without reranking due to error):")
    else:
        # Get final results without reranking
        final_results = combined_results.sort_values('score', ascending=False).head(top_k)
        print("Search results (without DeepSeek reranking):")
    
    # Print final results
    print(final_results[['product_id', 'price_usd', 'score']])
    
    return final_results

# Keep the original search_multi_modal function for backward compatibility
def search_multi_modal(query, text_index, image_index, df, model, processor, tfidf=None, tfidf_matrix=None, device="cpu", top_k=5, exchange_rate=86):
    print("Running original search_multi_modal function")
    
    # Generate query embedding
    inputs = processor(text=[query], return_tensors="pt").to(device)
    query_embedding = model.get_text_features(**inputs).detach().cpu().numpy()[0]
    query_embedding /= np.linalg.norm(query_embedding)

    # Search indices with larger candidate pool
    text_scores, text_indices = text_index.search(np.expand_dims(query_embedding, 0), top_k * 50)
    text_results = df.iloc[text_indices[0]].copy()
    text_results['text_score'] = text_scores[0]
    
    print("Initial text indices:", text_indices[0])
    print("B08CF3B7N1 in text results:", 'B08CF3B7N1' in df.iloc[text_indices[0]]['product_id'].values)
    
    target_idx = df[df['product_id'] == 'B08CF3B7N1'].index[0]
    if target_idx in text_indices[0]:
        rank = list(text_indices[0]).index(target_idx) + 1
        print(f"Initial rank of B08CF3B7N1: {rank}")

    image_scores, image_indices = image_index.search(np.expand_dims(query_embedding, 0), top_k * 50)
    image_results = df.iloc[image_indices[0]].copy()
    image_results['image_score'] = image_scores[0]

    # Combine results
    combined_results = pd.concat([text_results, image_results]).drop_duplicates(subset='product_id')
    combined_results['price_usd'] = pd.to_numeric(
        combined_results['discounted_price'].str.replace('₹', '').str.replace(',', ''),
        errors='coerce'
    ) / exchange_rate
    combined_results = combined_results.dropna(subset=['price_usd'])

    # Price filtering
    price_match = re.search(r'under (\d+(\.\d+)?)\s*USD', query, re.IGNORECASE)
    max_price = float(price_match.group(1)) if price_match else None
    if max_price:
        combined_results = combined_results[combined_results['price_usd'] < max_price]

    # Base scoring
    combined_results['score'] = (combined_results['text_score'].fillna(0) * 1.5 + 
                                 combined_results['image_score'].fillna(0) * 0.5)

    # Category detection
    query_lower = query.lower()
    if any(word in query_lower for word in ["cable", "charger", "cord"]) and "iphone" in query_lower:
        relevant_categories = ["USBCables"]
    elif any(word in query_lower for word in ["headset", "headphones", "earphones"]) and "noise cancelling" in query_lower:
        relevant_categories = ["PCHeadsets"]
    else:
        relevant_categories = []
    print("Relevant categories:", relevant_categories)

    # Debug target products
    target_ids = ['B08CF3B7N1', 'B009LJ2BXA']
    print("Before boosting:")
    print(combined_results[combined_results['product_id'].isin(target_ids)][['product_id', 'score']])

    # Apply category boost (additive)
    category_boost = 2.0
    for index, row in combined_results.iterrows():
        if any(cat.lower() in row['category'].lower() for cat in relevant_categories):
            combined_results.at[index, 'score'] += category_boost

    # Apply TF-IDF keyword boosting if available
    if tfidf is not None and tfidf_matrix is not None:
        query_keywords = tfidf.transform([query])
        keyword_scores = np.dot(tfidf_matrix[combined_results.index], query_keywords.T).toarray().ravel()
        combined_results['score'] += keyword_scores * 1.5  # Adjust multiplier as needed
        
        print("TF-IDF keyword score for B08CF3B7N1:", keyword_scores[combined_results['product_id'] == 'B08CF3B7N1'])

    # Apply price normalization if max_price exists
    if max_price:
        combined_results['price_score'] = 1 - (combined_results['price_usd'] / max_price).clip(0, 1)
        combined_results['score'] += combined_results['price_score'] * 0.5

    # Debug after boosting
    print("After boosting:")
    print(combined_results[combined_results['product_id'].isin(target_ids)][['product_id', 'score']])

    # Sort and return top results
    final_results = combined_results.sort_values('score', ascending=False).head(top_k)
    print("Search results:\n", final_results[['product_id', 'price_usd', 'score']])
    return final_results

def build_combined_embeddings(df, text_embeddings, img_dim=512, device="cpu"):
    product_urls = df["product_link"].tolist()
    image_embeddings_list = batch_get_image_embeddings(product_urls, batch_size=32, max_workers=20, device=device)
    image_embeddings = np.zeros((len(df), img_dim))
    for i, emb in enumerate(image_embeddings_list):
        if emb is not None:
            image_embeddings[i] = emb

    # Normalize embeddings
    text_mean = np.mean(text_embeddings, axis=0)
    text_std = np.std(text_embeddings, axis=0)
    text_std[text_std == 0] = 1
    norm_text = (text_embeddings - text_mean) / text_std
    norm_text = np.nan_to_num(norm_text)

    img_mean = np.mean(image_embeddings, axis=0)
    img_std = np.std(image_embeddings, axis=0)
    img_std[img_std == 0] = 1
    norm_image = (image_embeddings - img_mean) / img_std
    norm_image = np.nan_to_num(norm_image)

    combined_embeddings = np.concatenate([norm_text, norm_image], axis=1)
    return np.nan_to_num(combined_embeddings)

if __name__ == "__main__":
    df = load_processed_data("data/processed/processed_products.csv")
    print("Loaded processed data with", len(df), "products.")

    texts = df["cleaned_text"].fillna("").astype(str).tolist()
    device = "cpu"  # Set explicitly for testing
    text_embeddings, model = generate_embeddings(texts, device=device)
    print("Generated text embeddings of shape:", text_embeddings.shape)

    combined_embeds = build_combined_embeddings(df, text_embeddings, img_dim=512, device=device)
    print("Combined embeddings shape:", combined_embeds.shape)

    index_combined = build_faiss_index(combined_embeds, nlist=100)
    print("FAISS index built with", index_combined.ntotal, "vectors.")

    # Example usage with CLIP model (you'd pass these from outside in practice)
    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    example_query = "pink t-shirt with monkey logo in m size"
    results = search_multi_modal(
        example_query,
        text_index=index_combined,
        image_index=index_combined,
        df=df,
        model=clip_model,
        processor=processor,
        device=device,
        top_k=5
    )