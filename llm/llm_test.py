from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="dummy",
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
    messages=[{"role": "user", "content": "수신 신규 고객 중 골드 고객만 찾는 SQL문 줘"}],
)

print(response.choices[0].message.content)