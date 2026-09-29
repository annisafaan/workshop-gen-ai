import os
from openai import OpenAI

# 1. Hubungkan OpenAI SDK ke server lokal Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

# 2. Contoh System Prompt Lemah (Vague, tanpa batasan jelas)[cite: 20]
WEAK_SYSTEM = "You are an AI assistant."

# 3. Contoh System Prompt Kuat (Peran, tugas, aturan, dan format spesifik)[cite: 20, 21]
STRONG_SYSTEM = """You are a senior Python engineer reviewing code for a production AI pipeline.

Your job:
- Identify bugs, security issues, and performance problems
- Suggest concrete improvements with code examples
- Explain WHY each issue matters

Rules:
- Be direct. Do not pad with compliments.
- If code is correct, say so briefly and move on.
- Always include the corrected code when suggesting a fix.

Format:
Return your review as a numbered list. Each item: Issue -> Impact -> Fix."""

# 4. Target kode Python yang akan di-review (mengandung celah keamanan)[cite: 21]
target_code = """Review this function:

def get_user(user_id):
    key = os.getenv('DB_KEY')
    result = requests.get(f'http://db/{user_id}?key={key}')
    return result.json()"""

messages = [{"role": "user", "content": target_code}]

print("--- MENJALANKAN DENGAN STRONG SYSTEM PROMPT ---")

# 5. Kirim permintaan ke model lokal qwen2.5:3b menggunakan Strong System Prompt
response = client.chat.completions.create(
    model="qwen2.5:3b",
    messages=[{"role": "system", "content": STRONG_SYSTEM}] + messages,
)

print(response.choices[0].message.content)