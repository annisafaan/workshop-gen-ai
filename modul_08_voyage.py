import os
import numpy as np
from dotenv import load_dotenv
import voyageai

# 1. Muat environment variables dari file .env
load_dotenv()

# 2. Inisialisasi klien Voyage AI menggunakan API key[cite: 22]
vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

# 3. Kumpulan teks dokumen yang akan di-embed
texts = [
    "What is RAG?",
    "Explain vector databases.",
    "Retrieval-Augmented Generation combines search with LLMs."
]

print("--- MENJALANKAN VOYAGE AI EMBEDDINGS ---")

# 4. Panggil model embed voyage-3[cite: 22]
result = vo.embed(
    texts,
    model="voyage-3",        # Model embedding standar yang direkomendasikan[cite: 22]
    input_type="document"    # Gunakan "document" untuk basis data teks, atau "query" untuk kata kunci pencarian[cite: 22]
)

# 5. Konversi hasil list ke numpy array 2D[cite: 22]
embeddings = np.array(result.embeddings, dtype=np.float32)

print(f"Shape       : {embeddings.shape}")      # Menampilkan dimensi vektor (misal: 3 teks, 1024 dimensi)[cite: 22]
print(f"Token usage : {result.total_tokens}")   # Total token yang dihitung oleh API[cite: 22]