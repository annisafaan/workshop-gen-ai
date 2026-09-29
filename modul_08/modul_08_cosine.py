import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def pairwise_similarity(matrix: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity for all rows in a matrix.
    Returns an (n, n) matrix. All rows must be non-zero.
    """
    # Normaliser d'abord toutes les lignes à une longueur unitaire (norme L2 = 1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = matrix / norms
    
    # Le produit scalaire des vecteurs normalisés donne la matrice de similarité cosinus
    return (normed @ normed.T).astype(np.float32)

# Exemple d'utilisation avec des vecteurs aléatoires
rng = np.random.default_rng(42)
vecs = rng.standard_normal((4, 8)).astype(np.float32)
sim_matrix = pairwise_similarity(vecs)

print("Pairwise similarities:")
for i in range(4):
    for j in range(i + 1, 4):
        print(f" vec[{i}] vs vec[{j}]: {sim_matrix[i, j]:.4f}")

# La diagonale est toujours de 1.0 car un vecteur est identique à lui-même
print(f"\nDiagonal: {np.diag(sim_matrix)}")