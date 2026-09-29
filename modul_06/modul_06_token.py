from openai import OpenAI

# 1. Hubungkan OpenAI SDK ke server lokal Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

# 2. Definisikan tabel harga per 1 Juta token (Cost Estimator)[cite: 13]
PRICING = {
    "qwen2.5:3b": {"input": 0.10, "output": 0.20},  # Estimasi lokal
    "gpt-4o": {"input": 2.50, "output": 10.00},  #[cite: 13]
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},  #[cite: 13]
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
  """Return estimated cost in USD."""
  if model not in PRICING:
    raise ValueError(f"Unknown model: {model}")
  p = PRICING[model]
  return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


# 3. Definisikan Batas Maksimal Konteks (Context Window Limits)[cite: 14]
CONTEXT_LIMITS = {
    "qwen2.5:3b": 32768,  # Kapasitas context window qwen2.5:3b
    "gpt-4o": 128_000,  #[cite: 14]
    "claude-sonnet-4-5": 200_000,  #[cite: 14]
}


def fits_in_context(
    model: str, token_count: int, reserve_for_output: int = 2048
) -> bool:
  """Memeriksa apakah jumlah token muat dalam context limit model."""
  limit = CONTEXT_LIMITS.get(model, 128_000)
  return token_count + reserve_for_output <= limit


# 4. Uji interaksi dengan model lokal untuk melihat usage token asli dari response
model_name = "qwen2.5:3b"
prompt_text = "Explain the transformer architecture briefly."

print(f"Mengirim prompt ke model {model_name}...")
response = client.chat.completions.create(
    model=model_name, messages=[{"role": "user", "content": prompt_text}]
)

# Ambil data pemakaian token dari respons API
usage = response.usage
input_tokens = usage.prompt_tokens
output_tokens = usage.completion_tokens
total_tokens = usage.total_tokens

print("\n--- Hasil Token Usage dari API ---")
print(f"Input Tokens  : {input_tokens}")
print(f"Output Tokens : {output_tokens}")
print(f"Total Tokens  : {total_tokens}")

# 5. Hitung Estimasi Biaya
cost = estimate_cost(
    model_name, input_tokens=input_tokens, output_tokens=output_tokens
)
print(f"\nEstimated Cost: ${cost:.6f}")  #[cite: 14]

# 6. Cek Batas Konteks
is_safe = fits_in_context(model_name, token_count=input_tokens)
print(f"Fits in context window? : {is_safe}")

print("\nJawaban Model:")
print(response.choices[0].message.content)