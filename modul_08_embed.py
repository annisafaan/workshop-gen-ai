import numpy as np
from openai import OpenAI

# 1. Hubungkan OpenAI SDK ke server lokal Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

# 2. Fungsi untuk menghasilkan embeddings dari daftar teks[cite: 32]
def embed(texts: list[str], model: str = "nomic-embed-text") -> np.ndarray:
    """Embed a list of texts. Returns array of shape (n, dim)."""
    # Memanggil endpoint embeddings dari API[cite: 32]
    response = client.embeddings.create(input=texts, model=model)
    
    # Urutkan berdasarkan index untuk memastikan urutan output sama persis dengan input[cite: 32]
    vectors = sorted(response.data, key=lambda e: e.index)
    
    # Kembalikan sebagai array Numpy 2D[cite: 32]
    return np.array([v.embedding for v in vectors], dtype=np.float32)

# 3. Kumpulan teks yang akan diubah menjadi vektor[cite: 32]
texts = [
    "Retrieval-Augmented Generation combines search with LLMs.",
    "RAG retrieves documents then generates an answer from them.",
    "The Eiffel Tower is in Paris.",
    "Python is a popular programming language.",
    "Fine-tuning trains a model on new data.",
]

print("--- MENJALANKAN EMBEDDING GENERATION ---")
print("Sedang memproses perubahan teks menjadi vektor array...")

# 4. Hasilkan embeddings dan cek dimensinya[cite: 32]
embeddings = embed(texts)
print(f"\nShape: {embeddings.shape}") 

# 5. Cek panjang/norm dari vektor pertama (idealnya mendekati 1.0 jika sudah L2-normalised)[cite: 33]
norm = np.linalg.norm(embeddings[0])
print(f"Norm of first vector: {norm:.4f}")