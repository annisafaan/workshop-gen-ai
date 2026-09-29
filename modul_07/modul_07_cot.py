from openai import OpenAI

# 1. Hubungkan OpenAI SDK ke server lokal Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

# 2. Perbandingan prompt langsung vs Chain-of-Thought (CoT)
DIRECT_PROMPT = (
    "If a model costs $3.00 per million input tokens and $15.00 per million"
    " output tokens, and a request uses 2,400 input tokens and 800 output"
    " tokens, what is the total cost in USD?"
)

COT_PROMPT = """If a model costs $3.00 per million input tokens and $15.00 per million output tokens, and a request uses 2,400 input tokens and 800 output tokens, what is the total cost in USD?

Think through this step by step before giving the final answer."""

# Daftar eksperimen untuk membandingkan Direct vs CoT
prompts_to_test = [("Direct", DIRECT_PROMPT), ("Chain-of-Thought", COT_PROMPT)]

print("--- MENJALANKAN CHAIN-OF-THOUGHT PROMPTING ---")

for label, prompt in prompts_to_test:
  resp = client.chat.completions.create(
      model="qwen2.5:3b",
      max_tokens=512,
      messages=[{"role": "user", "content": prompt}],
  )

  print(f"=== {label} ===")
  print(resp.choices[0].message.content[:400])  # Cetak sebagian besar respons
  print("\n" + "=" * 40 + "\n")