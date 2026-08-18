import urllib.request
import json
import time

url = "https://api.semanticscholar.org/graph/v1/paper/ARXIV:1706.03762?fields=title,references.externalIds,references.title"
headers = {"User-Agent": "AcademicResearchBot/1.0 (mailto:academic@university.edu)"}

req = urllib.request.Request(url, headers=headers)
try:
    with urllib.request.urlopen(req, timeout=10) as response:
        data = json.loads(response.read().decode("utf-8"))
        print("S2 API SUCCESS for 1706.03762:")
        print("Title:", data.get("title"))
        refs = data.get("references", [])
        print("References count:", len(refs))
        if refs:
            print("First reference sample:", refs[0])
except Exception as e:
    print("S2 API Error:", e)
