from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="dummy",
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    messages=[{"role": "user", "content": "파이썬으로 피보나치 짜줘"}],
)

print(response.choices[0].message.content)