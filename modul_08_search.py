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

# 2. Struktur Data untuk Dokumen dan Hasil Pencarian[cite: 13]
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

# 3. Helper untuk membuat batch embeddings menggunakan Voyage AI
def embed_batch(texts: list[str], model: str = "voyage-3", input_type: str = "document") -> np.ndarray:
    """Embed texts in a single API call. Returns (n, dim) float32 array."""
    result = vo.embed(texts, model=model, input_type=input_type)
    return np.array(result.embeddings, dtype=np.float32)

# 4. Implementasi In-Memory Vector Store[cite: 14, 15, 16]
class VectorStore:
    """
    In-memory vector store for semantic search.
    Suitable for corpora up to ~100k documents.
    """
    def __init__(self, embed_model: str = "voyage-3"):
        self.embed_model = embed_model
        self._documents: list[Document] = []
        self._matrix: Optional[np.ndarray] = None  # (n, dim) normalised matrix

    def add_documents(self, documents: list[Document]) -> None:
        """Embed and index a list of documents."""
        texts = [d.text for d in documents]
        vectors = embed_batch(texts, model=self.embed_model, input_type="document")

        # Normalisasi vektor untuk mempercepat perhitungan cosine similarity via dot product[cite: 15]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = (vectors / norms).astype(np.float32)

        for doc, vec in zip(documents, normed):
            doc.embedding = vec
            self._documents.append(doc)

        # Gabungkan ke dalam satu matriks numpy[cite: 15]
        self._matrix = np.array(
            [d.embedding for d in self._documents], dtype=np.float32
        )
        print(f"Index successfully contains {len(self._documents)} documents.")

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        """Return the k most similar documents for a query string."""
        if self._matrix is None or len(self._documents) == 0:
            raise RuntimeError("No documents indexed yet.")

        # Embed dan normalisasi query[cite: 15, 16]
        q_vec = embed_batch([query], model=self.embed_model, input_type="query")[0]
        
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
        q_vec = (q_vec / q_norm).astype(np.float32)

        # Hitung kemiripan pakai Dot Product (matriks @ vektor query)[cite: 16]
        scores = self._matrix @ q_vec

        # Ambil top-k indeks tertinggi secara descending[cite: 16]
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

    @property
    def size(self) -> int:
        return len(self._documents)


# 5. Korpus Dokumen Contoh (Dataset Dummy)[cite: 17]
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

print("--- INITIALIZING VECTOR STORE & INDEXING CORPUS ---")
store = VectorStore()
store.add_documents(CORPUS)

# Jeda setelah indexing untuk mencegah bentrok rate limit
print("Menunggu sebentar setelah indexing...")
time.sleep(25)

# 6. Skenario Uji Pertanyaan (Queries)
QUERIES = [
    "How does RAG work?",
    "What algorithms do vector databases use?",
    "How do I split documents for embedding?",
]

print("\n--- RUNNING SEMANTIC SEARCH QUERIES ---")
for query in QUERIES:
    print(f"\nQuery: {query!r}")
    results = store.search(query, k=3)
    for r in results:
        print(f"  [{r.rank}] score={r.score:.4f} | {r.document.text[:80]}...")
    
    # Berikan jeda 25 detik antar kueri agar aman dari batas 3 RPM
    print("Menunggu sebentar untuk menghindari rate limit...")
    time.sleep(25)