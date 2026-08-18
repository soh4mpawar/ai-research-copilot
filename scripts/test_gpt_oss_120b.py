import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq

client = Groq(api_key=os.getenv('GROQ_API_KEY'))
try:
    resp = client.chat.completions.create(
        messages=[{'role': 'user', 'content': 'Evaluate this answer: The transformer uses self-attention. Return JSON: {"score": 1}'}],
        model='openai/gpt-oss-120b'
    )
    print('Groq openai/gpt-oss-120b SUCCESS:')
    print('Content:', resp.choices[0].message.content)
    print('Usage:', resp.usage)
except Exception as e:
    print('Groq openai/gpt-oss-120b ERROR:', e)
