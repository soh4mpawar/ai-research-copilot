import urllib.request
import json

papers = [
    ("2608.12841", "Attention"),
    ("2608.12626", "Retrieval-Augmented Generation")
]

for aid, target_title in papers:
    url = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{aid}?fields=title,references.title,references.externalIds"
    headers = {"User-Agent": "AcademicBot/1.0"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            d = json.loads(resp.read().decode("utf-8"))
            t = d.get("title")
            print(f"Paper {aid}: {t}")
            matched = [r for r in d.get("references", []) if target_title.lower() in (r.get("title") or "").lower()]
            for m in matched:
                ref_t = m.get("title")
                ext = m.get("externalIds")
                print(f"  -> Ground Truth Reference: '{ref_t}' | ExternalIDs: {ext}")
    except Exception as e:
        print(f"Error {aid}: {e}")
