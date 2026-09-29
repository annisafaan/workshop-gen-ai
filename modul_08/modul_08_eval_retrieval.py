import os
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
import voyageai

# 1. Muat environment variables dan inisialisasi Voyage AI
load_dotenv()
vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

@dataclass
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

@dataclass
class SearchResult:
    document: Document
    score: float
    rank: int

def embed_batch(texts: list[str], model: str = "voyage-3", input_type: str = "document") -> np.ndarray:
    result = vo.embed(texts, model=model, input_type=input_type)
    return np.array(result.embeddings, dtype=np.float32)

class VectorStore:
    def __init__(self, embed_model: str = "voyage-3"):
        self.embed_model = embed_model
        self._documents: list[Document] = []
        self._matrix: Optional[np.ndarray] = None

    def add_documents(self, documents: list[Document]) -> None:
        texts = [d.text for d in documents]
        vectors = embed_batch(texts, model=self.embed_model, input_type="document")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = (vectors / norms).astype(np.float32)

        for doc, vec in zip(documents, normed):
            doc.embedding = vec
            self._documents.append(doc)

        self._matrix = np.array([d.embedding for d in self._documents], dtype=np.float32)

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        if self._matrix is None or len(self._documents) == 0:
            raise RuntimeError("No documents indexed yet.")
        
        q_vec = embed_batch([query], model=self.embed_model, input_type="query")[0]
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
        q_vec = (q_vec / q_norm).astype(np.float32)

        scores = self._matrix @ q_vec
        k = min(k, len(self._documents))
        top_idx = np.argsort(scores)[::-1][:k]

        return [
            SearchResult(
                document=self._documents[int(i)],
                score=float(scores[i]),
                rank=rank + 1,
            )
            for rank, i in enumerate(top_idx)
        ]

# 2. Struktur Data dan Fungsi Evaluasi Retrieval
@dataclass
class RetrievalEvalCase:
    query: str
    relevant_doc_ids: list[str]  # ground-truth relevant documents

def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k

def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of all relevant docs found in top-k."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)

def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """MRR: reciprocal of the rank of the first relevant result."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0

def evaluate_retrieval(store: VectorStore, eval_cases: list[RetrievalEvalCase], k: int = 3) -> dict:
    """Run all eval cases and return aggregate metrics."""
    p_scores, r_scores, mrr_scores = [], [], []

    for case in eval_cases:
        results = store.search(case.query, k=k)
        retrieved_ids = [r.document.id for r in results]

        p_scores.append(precision_at_k(retrieved_ids, case.relevant_doc_ids, k))
        r_scores.append(recall_at_k(retrieved_ids, case.relevant_doc_ids, k))
        mrr_scores.append(mean_reciprocal_rank(retrieved_ids, case.relevant_doc_ids))
        
        # Berikan jeda antar kueri evaluasi agar aman dari rate limit
        time.sleep(22)

    return {
        f"precision@{k}": round(float(np.mean(p_scores)), 4),
        f"recall@{k}": round(float(np.mean(r_scores)), 4),
        "MRR": round(float(np.mean(mrr_scores)), 4),
    }

# 3. Korpus Dokumen dan Kasus Uji Evaluasi
CORPUS = [
    Document("d01", "Retrieval-Augmented Generation (RAG) combines information retrieval with language model generation to answer questions using external knowledge."),
    Document("d02", "Vector databases store high-dimensional embeddings and enable fast approximate nearest-neighbour search using algorithms like HNSW and IVF."),
    Document("d03", "Fine-tuning adapts a pre-trained language model to a specific task by continuing training on a curated dataset with task-specific examples."),
    Document("d04", "Prompt engineering involves designing and optimising input prompts to guide language models toward producing the desired output."),
    Document("d05", "LangChain is a Python framework that provides abstractions for building applications with large language models, including chains, agents, and memory."),
    Document("d06", "Cosine similarity measures the angle between two vectors and is the standard metric for comparing text embeddings in semantic search."),
    Document("d07", "RLHF (Reinforcement Learning from Human Feedback) aligns language models with human preferences by training a reward model on human rankings."),
    Document("d08", "Chunking strategies for RAG include fixed-size chunks, sentence-aware splits, and recursive character splitting with configurable overlap."),
    Document("d09", "The transformer architecture uses self-attention mechanisms to model relationships between all tokens in a sequence simultaneously."),
    Document("d10", "Agents use language models as a reasoning engine, enabling them to plan multi-step tasks, call tools, and take actions based on observations."),
]

eval_cases = [
    RetrievalEvalCase("How does RAG work?", ["d01", "d08"]),
    RetrievalEvalCase("What are vector databases?", ["d02", "d06"]),
    RetrievalEvalCase("How do agents use language models?", ["d10"]),
    RetrievalEvalCase("What is fine-tuning?", ["d03"]),
    RetrievalEvalCase("How do transformers model token relationships?", ["d09"]),
]

print("--- INITIALIZING VECTOR STORE & INDEXING CORPUS ---")
store = VectorStore()
store.add_documents(CORPUS)
print("Indeks dokumen selesai. Menunggu sebentar sebelum evaluasi...")
time.sleep(25)

print("\n--- MENJALANKAN RETRIEVAL EVALUATION ---")
metrics = evaluate_retrieval(store, eval_cases, k=3)

print("\nHasil Evaluasi Retrieval:")
print(metrics)