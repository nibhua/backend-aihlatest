# indexer/

Vector indexing and similarity search functionality for document retrieval.

## What it does

- **Vector Storage**: Manages vector embeddings for document chunks
- **Similarity Search**: Performs nearest neighbor search using FAISS
- **Collection Management**: Handles document collection creation and indexing
- **Reranking**: Post-processing search results for improved relevance
- **Fusion**: Combines multiple search strategies for better results

## Files

- `nn_store.py` — Core vector storage and similarity search implementation
- `config.py` — Indexer configuration and settings
- `rerank.py` — Search result reranking and post-processing
- `fusion.py` — Multi-strategy search fusion
- `__init__.py` — Module exports (NNStore, make_collection_id)

## Key Features

- **FAISS Integration**: High-performance vector similarity search
- **Collection IDs**: Unique identifiers for document collections
- **Reranking**: Improves search result quality through post-processing
- **Fusion**: Combines different search approaches for optimal results

## Usage

```python
from indexer import NNStore, make_collection_id

# Create collection
collection_id = make_collection_id()
store = NNStore(collection_id)

# Add vectors and perform search
store.add_vectors(embeddings, metadata)
results = store.search(query_vector, k=10)
```
