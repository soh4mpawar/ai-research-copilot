import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
try:
    resp = client.chat.completions.create(
        messages=[{"role": "user", "content": "Evaluate this answer: The transformer uses self-attention. Return JSON: {\"score\": 1}"}],
        model="gpt-4o-mini"
    )
    print("OpenAI gpt-4o-mini SUCCESS:")
    print("Content:", resp.choices[0].message.content)
    print("Usage:", resp.usage)
except Exception as e:
    print("OpenAI gpt-4o-mini ERROR:", e)
