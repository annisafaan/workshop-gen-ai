import json
from dataclasses import dataclass
from openai import OpenAI

# 1. Hubungkan OpenAI SDK ke server lokal Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

# 2. Definisikan struktur data untuk skenario uji[cite: 29, 30]
@dataclass
class EvalCase:
    input_text: str
    expected_keywords: list[str]  # Jawaban yang diharapkan muncul[cite: 30]
    must_be_json: bool = False

# 3. Fungsi untuk mengevaluasi prompt secara otomatis[cite: 30]
def evaluate_prompt(system: str, cases: list[EvalCase]) -> dict:
    """Run a prompt against test cases and return pass rate + details."""
    results = []

    for case in cases:
        resp = client.chat.completions.create(
            model="qwen2.5:3b",
            max_tokens=256,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": case.input_text}
            ],
        )
        text = resp.choices[0].message.content.strip()

        # Check keyword hit (apakah keyword yang diharapkan ada di dalam respons)[cite: 30]
        keyword_hit = any(kw.lower() in text.lower() for kw in case.expected_keywords)

        # Check JSON validity jika diwajibkan[cite: 30]
        json_valid = True
        if case.must_be_json:
            try:
                json.loads(text)
            except json.JSONDecodeError:
                json_valid = False

        # Tentukan status lulus (passed)[cite: 30]
        passed = keyword_hit and json_valid
        results.append({
            "input": case.input_text[:60],
            "passed": passed,
            "response_preview": text[:80],
        })

    # Hitung persentase kelulusan[cite: 31]
    pass_rate = sum(r["passed"] for r in results) / len(results)
    return {"pass_rate": pass_rate, "results": results}


# 4. Siapkan Prompt yang akan dievaluasi (Tugas Klasifikasi)[cite: 31]
CLASSIFY_SYSTEM = """Classify the AI task as one of: CLASSIFICATION, GENERATION, RETRIEVAL, EMBEDDING.
Return ONLY the category word."""

# 5. Siapkan daftar soal uji (Test Cases)[cite: 31]
test_cases = [
    EvalCase("Predict whether an email is spam.", ["CLASSIFICATION"]),
    EvalCase("Write a product description for headphones.", ["GENERATION"]),
    EvalCase("Find the most relevant documents for a query.", ["RETRIEVAL"]),
    EvalCase("Convert this sentence to a vector.", ["EMBEDDING"]),
    EvalCase("Label customer reviews as positive or negative.", ["CLASSIFICATION"]),
]

print("--- MENJALANKAN PROMPT EVALUATION ---")
print("Mengevaluasi 5 test cases...")

# 6. Jalankan evaluasi dan cetak laporannya[cite: 31]
report = evaluate_prompt(CLASSIFY_SYSTEM, test_cases)

print(f"\nPass rate: {report['pass_rate']:.0%}")
for r in report["results"]:
    status = "PASS" if r["passed"] else "FAIL"
    print(f"[{status}] {r['input']!r} -> {r['response_preview']!r}")