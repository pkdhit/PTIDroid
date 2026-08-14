from openai import OpenAI
import time

client = OpenAI(
    base_url="YOUR_BASE_URL",
    api_key="YOUR_API_KEY"
)

start = time.time()
response = client.chat.completions.create(
    model="model",
    messages=[
        {"role": "user", "content": "Hello, please say 'OK'"}
    ],
    temperature=0.2,
    max_tokens=10,
    timeout=60
)
elapsed = time.time() - start
print(f"Response: {response.choices[0].message.content}")
print(f"Time: {elapsed:.2f}s")