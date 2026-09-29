from openai import OpenAI

# 1. Hubungkan OpenAI SDK ke server lokal Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

# 2. Definisikan system prompt beserta beberapa contoh (Few-shot examples)[cite: 23]
FEW_SHOT_SYSTEM = """You are a data extractor. Given a raw AI benchmark result string,
extract: model name, task, and score as a JSON object.

Examples:

Input: "GPT-4o scored 87.3% on the MMLU science subset"
Output: {"model": "gpt-4o", "task": "MMLU science", "score": 87.3}

Input: "Claude Sonnet 4.5 achieved 92.1 on HumanEval"
Output: {"model": "claude-sonnet-4-5", "task": "HumanEval", "score": 92.1}

Input: "Gemini 1.5 Pro: 78.9% accuracy on GSM8K math"
Output: {"model": "gemini-1.5-pro", "task": "GSM8K math", "score": 78.9}

Return ONLY the JSON object. No explanation."""

# 3. Kumpulan data uji yang akan diekstrak oleh model[cite: 23]
test_inputs = [
    "GPT-4o-mini reached 82.0% on MMLU",
    "Llama 3.1 70B: 86.4 on TruthfulQA",
    "Claude Opus 4.5 scored 96.7 on SWE-bench Verified",
]

print("--- MENJALANKAN FEW-SHOT PROMPTING ---")

# 4. Loop untuk menguji setiap input menggunakan qwen2.5:3b
for text in test_inputs:
  resp = client.chat.completions.create(
      model="qwen2.5:3b",
      max_tokens=128,
      messages=[
          {"role": "system", "content": FEW_SHOT_SYSTEM},
          {"role": "user", "content": text},
      ],
  )

  print(f"Input : {text}")  #[cite: 24]
  print(f"Output: {resp.choices[0].message.content}\n")  #[cite: 24]