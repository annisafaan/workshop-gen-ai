from abc import ABC, abstractmethod
from dataclasses import dataclass
from openai import OpenAI


@dataclass
class ChatMessage:
  role: str  # "user" atau "assistant"
  content: str


@dataclass
class ChatResponse:
  text: str
  input_tokens: int
  output_tokens: int
  model: str


# 1. Base Class (Abstraksi Klien LLM)
class BaseLLMClient(ABC):

  @abstractmethod
  def chat(
      self,
      messages: list[ChatMessage],
      system: str = "",
      max_tokens: int = 1024,
      temperature: float = 0.7,
  ) -> ChatResponse:
    pass


# 2. Implementasi Klien untuk Ollama (Lokal)
class OllamaClient(BaseLLMClient):

  def __init__(self, model: str = "qwen2.5:3b"):
    self.model = model
    self._client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # Ollama lokal tidak butuh API key asli
    )

  def chat(
      self,
      messages: list[ChatMessage],
      system: str = "",
      max_tokens: int = 1024,
      temperature: float = 0.7,
  ) -> ChatResponse:
    api_messages = []
    if system:
      api_messages.append({"role": "system", "content": system})

    for m in messages:
      api_messages.append({"role": m.role, "content": m.content})

    resp = self._client.chat.completions.create(
        model=self.model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=api_messages,
    )

    return ChatResponse(
        text=resp.choices[0].message.content,
        input_tokens=resp.usage.prompt_tokens,
        output_tokens=resp.usage.completion_tokens,
        model=self.model,
    )


# 3. Contoh Penggunaan (Provider-Agnostic Pipeline)
if __name__ == "__main__":
  # Kita bisa mengganti implementasi klien dengan mudah di sini
  client: BaseLLMClient = OllamaClient(model="qwen2.5:3b")

  msgs = [ChatMessage(role="user", content="What is a vector database?")]
  result = client.chat(msgs, system="Be concise.")

  print("--- Hasil Respon Model ---")
  print(result.text)
  print(
      f"\nToken Usage -> Input: {result.input_tokens} tokens, Output:"
      f" {result.output_tokens} tokens"
  )
  print(f"Model Used  -> {result.model}")