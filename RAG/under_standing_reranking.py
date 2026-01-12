from dataclasses import dataclass
from typing import List, Tuple
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class Document:
    content: str
    embedding: np.ndarray
    intial_score: float # Fixed typo here
    metadata: dict = None

def rerank_documents(
        query_embeddings: np.ndarray,
        documents: List[Document],
        semantic_weights: float = 0.5,
        initial_weight: float = 0.5
        ) -> List[Tuple[Document, float]]:
    
    total_weight = semantic_weights + initial_weight
    semantic_weights = semantic_weights / total_weight
    initial_weight = initial_weight / total_weight

    # stack embedding into matrix (N, dimensions)
    doc_embedding = np.vstack([doc.embedding for doc in documents])

    # Fix: Ensure query is (1, dimensions) to match doc_embedding (N, dimensions)
    # Since query_embeddings is already (1, 3) from the caller, we use it directly
    semantic_scores = cosine_similarity(
        query_embeddings, 
        doc_embedding
    )[0]

    # get initial scores and normalize to 0-1
    initial_scores = np.array([doc.intial_score for doc in documents])
    
    # Handle edge case: if all scores are the same, avoid division by zero
    def normalize(array):
        denom = array.max() - array.min()
        return (array - array.min()) / denom if denom > 0 else np.zeros_like(array)

    semantic_scores = normalize(semantic_scores)
    initial_scores = normalize(initial_scores)

    final_score = (semantic_weights * semantic_scores) + (initial_weight * initial_scores)

    ranked_result = list(zip(documents, final_score))
    ranked_result.sort(key=lambda x: x[1], reverse=True)

    return ranked_result

def example_reranking():
    documents = [
        Document("the quick brown fox", np.array([0.1, 0.2, 0.3]), 0.8),
        Document("Jumps over the lazy dog", np.array([0.1, 0.2, 0.3]), 0.6),
        Document("The dog sleeps peacefully", np.array([0.1, 0.2, 0.3]), 0.9)
    ]

    # Query embedding as (1, 3)
    query_embedding = np.array([0.15, 0.25, 0.35]).reshape(1, -1)

    reranked_docs = rerank_documents(
        query_embeddings=query_embedding,
        documents=documents,
        semantic_weights=0.7,
        initial_weight=0.3
    )

    print("\nReranked Documents:")
    for doc, score in reranked_docs:
        # Added the 'f' prefix for the f-string
        print(f"Score: {score:.3f} - Content: {doc.content}")

if __name__ == "__main__":
    example_reranking()