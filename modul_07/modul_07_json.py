import json
from openai import OpenAI

# 1. Hubungkan OpenAI SDK ke server lokal Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

# 2. Siapkan System Prompt berisi skema JSON yang diinginkan
SYSTEM_PROMPT = """Extract entities from the text. 
Return ONLY a JSON object with this exact schema:
{"people": [string], "organizations": [string], "locations": [string]}"""

user_text = "Elon Musk founded SpaceX in Hawthorne, California. He also leads Tesla."

print("--- MENJALANKAN STRUCTURED OUTPUT (JSON MODE) ---")

# 3. Panggil API dengan parameter response_format={"type": "json_object"}
response = client.chat.completions.create(
    model="qwen2.5:3b",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text}
    ],
)

# 4. Ambil dan parse string JSON menjadi dictionary Python
raw_content = response.choices[0].message.content
print("Raw Response String:")
print(raw_content)

try:
  # Konversi string JSON ke dictionary
  parsed_data = json.loads(raw_content)
  
  print("\nParsed Python Dictionary:")
  # Cetak dengan format rapi (indent=2) agar mudah dibaca
  print(json.dumps(parsed_data, indent=2))
  
except json.JSONDecodeError as e:
  print(f"\nError: Model tidak mengembalikan JSON yang valid. {e}")