from dataclasses import dataclass
import re

# 1. Struktur Data untuk Chunk
@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int

# 2. Fungsi untuk memecah teks berdasarkan kalimat dan ukuran maksimal
def chunk_by_sentences(
    text: str,
    doc_id: str,
    max_chars: int = 1000,
    overlap_chars: int = 100,
) -> list[Chunk]:
    """
    Split text into chunks that respect sentence boundaries.
    Adds overlap so context is not lost at chunk edges.
    """
    # Memisahkan teks berdasarkan akhir kalimat (. ! ?)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks: list[Chunk] = []
    current = ""
    char_offset = 0
    chunk_idx = 0

    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence

        if len(candidate) > max_chars and current:
            # Simpan chunk saat ini
            end = char_offset + len(current)
            chunks.append(Chunk(doc_id, chunk_idx, current.strip(), char_offset, end))
            chunk_idx += 1

            # Mulai chunk baru dengan menyertakan overlap dari akhir chunk sebelumnya
            overlap_start = max(0, len(current) - overlap_chars)
            overlap_text = current[overlap_start:]
            current = (overlap_text + " " + sentence).strip()
            char_offset = end - len(overlap_text)
        else:
            current = candidate

    # Simpan chunk terakhir
    if current.strip():
        end = char_offset + len(current)
        chunks.append(Chunk(doc_id, chunk_idx, current.strip(), char_offset, end))

    return chunks

# 3. Uji coba dengan dokumen sampel
document = """
Large language models (LLMs) are neural networks trained on vast amounts of text data.
They learn to predict the next token in a sequence, which gives them broad language understanding.
Models like GPT-4 and Claude are examples of LLMs used in production today.

Retrieval-Augmented Generation, or RAG, extends LLMs by connecting them to external knowledge bases.
Instead of relying solely on knowledge encoded during training, a RAG system retrieves relevant documents at inference time.
This allows the model to answer questions about recent events or private data it was never trained on.

The retrieval step in RAG typically uses embedding-based semantic search.
A query is embedded into a vector, and the nearest document vectors are retrieved from a database.
These documents are then injected into the context window alongside the query.
"""

print("--- MENJALANKAN TEXT CHUNKING ---")
# Memecah dokumen dengan batas maksimal 300 karakter dan overlap 50 karakter
chunks = chunk_by_sentences(document.strip(), doc_id="intro_to_llms", max_chars=300, overlap_chars=50)

print(f"Total chunk yang dihasilkan: {len(chunks)}\n")
for c in chunks:
    print(f"Chunk {c.chunk_index} (Karakter {c.char_start}-{c.char_end}):\n{c.text}\n{'-'*40}")