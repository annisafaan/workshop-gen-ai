from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key=os.environ["OPENAI_API_KEY"]
)

stream = client.chat.completions.create(
    model="qwen2.5vl:3b",
    max_tokens=512,
    stream=True,
    messages=[{"role": "user", "content": "Explain embeddings in 3 bullet points."}]
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)

print()