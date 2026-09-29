import json
from openai import OpenAI

# 1. Hubungkan OpenAI SDK ke server lokal Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama tidak memerlukan API key asli
)

# 2. Definisikan skema tool (fungsi yang bisa dipanggil oleh model)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_model_info",
            "description": (
                "Returns context window and pricing for a given LLM."
            ),  #[cite: 7]
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Model identifier.",  #[cite: 7]
                    }
                },
                "required": ["model_name"],  #[cite: 7]
            },
        },
    }
]


# 3. Fungsi asli yang akan dieksekusi di sisi Python[cite: 7]
def get_model_info(model_name: str) -> dict:
  db = {
      "gpt-4o": {"context_k": 128, "cost_input": 2.50},  #[cite: 7]
      "claude-sonnet-4-5": {"context_k": 200, "cost_input": 3.00},  #[cite: 7]
  }
  return db.get(model_name, {"error": "unknown model"})  #[cite: 7]


# 4. Tambahkan system prompt agar model wajib menggunakan tool
messages = [
    {
        "role": "system",
        "content": (
            "You must use the get_model_info tool to answer questions about"
            " model specifications."
        ),
    },
    {"role": "user", "content": "What is gpt-4o's context window?"},
]

# 5. Pemanggilan API pertama menggunakan qwen2.5:3b
response = client.chat.completions.create(
    model="qwen2.5:3b",  # Menggunakan model teks yang support tools
    tools=tools,
    messages=messages,
)

# 6. Cek apakah model ingin memanggil tool[cite: 7]
if response.choices[0].finish_reason == "tool_calls":
  tool_call = response.choices[0].message.tool_calls[0]
  name = tool_call.function.name  #[cite: 7]
  args = json.loads(tool_call.function.arguments)  #[cite: 7]
  result = get_model_info(**args)  #[cite: 7]

  print(f"[Tool Called]: {name}({args})")
  print(f"[Tool Result]: {result}")

  # Masukkan respons asisten dan hasil tool ke riwayat pesan[cite: 8]
  messages.append(response.choices[0].message)
  messages.append({
      "role": "tool",
      "tool_call_id": tool_call.id,
      "content": json.dumps(result),
  })

  # 7. Pemanggilan API terakhir untuk mendapatkan jawaban akhir dari model[cite: 8]
  final = client.chat.completions.create(model="qwen2.5:3b", messages=messages)

  print("\nFinal Answer:")
  print(final.choices[0].message.content)  #[cite: 8]
else:
  print("\nModel menjawab langsung:")
  print(response.choices[0].message.content)