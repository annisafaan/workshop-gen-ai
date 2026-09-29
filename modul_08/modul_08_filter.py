import os
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from dotenv import load_dotenv
import voyageai

# 1. Muat environment variables dan inisialisasi Voyage AI
load_dotenv()
vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

@dataclass
class FilteredDocument:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

def embed_texts(texts: list[str], model: str = "voyage-3", input_type: str = "document") -> np.ndarray:
    """Embed texts menggunakan Voyage AI dan kembalikan numpy array float32."""
    result = vo.embed(texts, model=model, input_type=input_type)
    return np.array(result.embeddings, dtype=np.float32)

class FilteredVectorStore:
    def __init__(self):
        self._docs: list[FilteredDocument] = []

    def add(self, docs: list[FilteredDocument]) -> None:
        """Menambahkan dan mengindeks dokumen beserta embed-nya."""
        texts = [d.text for d in docs]
        embeddings = embed_texts(texts, input_type="document")
        
        # Normalisasi L2 untuk persiapan dot product cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = (embeddings / norms).astype(np.float32)

        for doc, emb in zip(docs, normed):
            doc.embedding = emb
            self._docs.append(doc)
        print(f"Berhasil mengindeks {len(docs)} dokumen.")

    def search(
        self,
        query: str,
        k: int = 5,
        filter_fn: Optional[Callable[[FilteredDocument], bool]] = None,
    ) -> list[tuple[FilteredDocument, float]]:
        """Search dengan opsi metadata filter yang diterapkan sebelum ranking."""
        # Terapkan pre-filter jika ada fungsi filter yang diberikan
        candidates = self._docs if filter_fn is None else [d for d in self._docs if filter_fn(d)]
        if not candidates:
            return []

        # Embed query menggunakan input_type="query"
        q_result = vo.embed([query], model="voyage-3", input_type="query")
        q_vec = np.array(q_result.embeddings[0], dtype=np.float32)
        
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        # Hitung skor kemiripan dengan matrix dot product
        matrix = np.array([d.embedding for d in candidates], dtype=np.float32)
        scores = matrix @ q_vec

        # Urutkan berdasarkan top-k skor tertinggi
        k = min(k, len(candidates))
        top_idx = np.argsort(scores)[::-1][:k]

        return [(candidates[i], float(scores[i])) for i in top_idx]

# 2. Demo - Korpus campuran dengan metadata kategori dan tahun
docs = [
    FilteredDocument("a1", "GPT-4o supports vision and function calling.", {"category": "openai", "year": 2024}),
    FilteredDocument("a2", "Claude 3.5 Sonnet excels at coding tasks.", {"category": "anthropic", "year": 2024}),
    FilteredDocument("a3", "GPT-4o-mini is a smaller, cheaper model.", {"category": "openai", "year": 2024}),
    FilteredDocument("a4", "Claude Opus 4 is Anthropic's most capable model.", {"category": "anthropic", "year": 2025}),
    FilteredDocument("a5", "GPT-4 Turbo has a 128K context window.", {"category": "openai", "year": 2023}),
]

print("--- INITIALIZING FILTERED VECTOR STORE ---")
fstore = FilteredVectorStore()
fstore.add(docs)

# Jeda setelah menambahkan dokumen untuk menghindari rate limit
print("Menunggu sebentar setelah add dokumen...")
time.sleep(25)

# 3. Pencarian di seluruh dokumen (Tanpa Filter)
print("\n=== All docs (Pencarian Global) ===")
results = fstore.search("which model is good at coding?", k=3)
for doc, score in results:
    print(f"  [{score:.4f}] {doc.id} ({doc.metadata['category']}): {doc.text}")

# Jeda sebelum kueri berikutnya
print("Menunggu sebentar untuk menghindari rate limit...")
time.sleep(25)

# 4. Pencarian dengan Filter Metadata (Hanya dokumen Anthropic)
print("\n=== Anthropic only (Pencarian dengan Filter) ===")
results = fstore.search(
    "which model is good at coding?",
    k=3,
    filter_fn=lambda d: d.metadata["category"] == "anthropic"
)
for doc, score in results:
    print(f"  [{score:.4f}] {doc.id} ({doc.metadata['category']}): {doc.text}")