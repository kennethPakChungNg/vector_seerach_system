"""
vectorshop.embedding package initialization.

Exports the main functions and classes for the vector search system.
"""

from .vector_search import (
    get_text_model,
    load_processed_data,
    generate_embeddings,
    build_faiss_index,
    improved_search_multi_modal,
    search_multi_modal,
    build_combined_embeddings,

)


from .deepseek_embeddings import (
    DeepSeekEmbeddings,
    create_product_text
)

from .bm25_search import (
    BM25,
    ProductBM25Search
)

from .hybrid_search import (
    HybridSearch
)

from .embedding_tracker import EmbeddingTracker

