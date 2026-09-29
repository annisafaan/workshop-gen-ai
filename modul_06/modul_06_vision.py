import base64
from openai import OpenAI

# 1. Hubungkan OpenAI SDK ke server lokal Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama tidak memerlukan API key asli
)


# 2. Fungsi untuk mendeskripsikan gambar lokal menggunakan Base64
def describe_image_file(image_path: str) -> str:
  # Baca file gambar lalu ubah ke format base64
  with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

  # Kirim gambar dan teks perintah ke model qwen2.5vl:3b
  response = client.chat.completions.create(
      model="qwen2.5vl:3b",  # Menggunakan model vision lokal
      messages=[{
          "role": "user",
          "content": [
              {
                  "type": "image_url",
                  "image_url": {
                      "url": f"data:image/jpeg;base64,{encoded_image}"
                  },
              },
              {
                  "type": "text",
                  "text": (
                      "Describe what you see in this image in detail."
                  ),
              },
          ],
      }],
  )
  return response.choices[0].message.content


# 3. Contoh Penggunaan
# (Pastikan kamu menaruh sebuah file gambar .jpg/.png di folder yang sama, lalu ganti namanya di bawah)
image_filename = "contoh_gambar.jpg"

print(f"Menganalisis gambar '{image_filename}'...")
try:
  result = describe_image_file(image_filename)
  print("\nHasil Analisis Model:")
  print(result)
except FileNotFoundError:
  print(
      f"\n[Perhatian]: File gambar '{image_filename}' tidak ditemukan di"
      " folder!"
  )
  print(
      "Silakan masukkan contoh file gambar berformat .jpg atau .png ke folder"
      " project ini dan ubah variabel image_filename."
  )