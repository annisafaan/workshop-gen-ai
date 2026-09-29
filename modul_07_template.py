from dataclasses import dataclass, field
from string import Formatter
from typing import Any
from openai import OpenAI

# 1. Hubungkan ke server lokal Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

# 2. Buat Class PromptTemplate[cite: 32, 33]
@dataclass
class PromptTemplate:
    """A reusable, versioned prompt template."""
    name: str
    system: str
    user: str
    version: str = "1.0"
    required_vars: list[str] = field(default_factory=list)

    def __post_init__(self):
        # Auto-detect required variables from both templates[cite: 33]
        formatter = Formatter()
        combined = self.system + self.user
        self.required_vars = [
            fname for _, fname, _, _ in formatter.parse(combined)
            if fname is not None
        ]

    def render(self, **kwargs: Any) -> tuple[str, str]:
        """Return (rendered_system, rendered_user). Raises if vars are missing."""
        # Mengecek apakah ada variabel yang kurang[cite: 33]
        missing = set(self.required_vars) - set(kwargs)
        if missing:
            raise ValueError(f"Missing template variables: {missing}")
        
        return self.system.format(**kwargs), self.user.format(**kwargs)

# 3. Definisikan template sebagai konstanta[cite: 33]
QA_TEMPLATE = PromptTemplate(
    name="question_answering",
    version="1.2",
    system="You are a {domain} expert. Answer questions accurately and concisely.\nCite sources when possible. If you are unsure, say so.",
    user="Question: {question}\n\nContext: {context}",
)

print("--- MENJALANKAN PROMPT TEMPLATE ---")

try:
    # 4. Penggunaan: Mengisi variabel ke dalam template[cite: 34]
    system_prompt, user_prompt = QA_TEMPLATE.render(
        domain="machine learning",
        question="What is the vanishing gradient problem?",
        context="Gradients in deep networks are computed via backpropagation...",
    )

    print("System Prompt Rendered:\n", system_prompt)
    print("\nUser Prompt Rendered:\n", user_prompt)
    print("\nRequired vars terdeteksi otomatis:", QA_TEMPLATE.required_vars)

    # 5. Mengirim prompt yang sudah di-render ke Ollama
    print("\n--- MENGIRIM KE MODEL OLLAMA ---")
    response = client.chat.completions.create(
        model="qwen2.5:3b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    print("\nJawaban Model:")
    print(response.choices[0].message.content)

except ValueError as e:
    print("\nTerjadi Error Template:", e)