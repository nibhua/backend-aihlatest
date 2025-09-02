from .nn_store import NNStore, make_collection_id

# Back-compat alias (some code may still import NNVectorStore)
NNVectorStore = NNStore

__all__ = ["NNStore", "NNVectorStore", "make_collection_id"]