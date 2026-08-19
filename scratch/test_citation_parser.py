import re

def format_citations(text: str) -> str:
    def replace_citation(match):
        inner = match.group(1)
        # Parse tokens separated by commas or semicolons
        tokens = [t.strip() for t in re.split(r'[,;]+', inner) if t.strip()]
        nums = []
        for tok in tokens:
            if '-' in tok or '–' in tok:
                parts = re.split(r'[-–]', tok)
                if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                    start, end = int(parts[0]), int(parts[1])
                    if 0 < end - start <= 10:
                        nums.extend([str(i) for i in range(start, end + 1)])
                        continue
            if tok.isdigit():
                nums.append(tok)
        if not nums:
            return match.group(0)
        chips = [f'<a href="#src-{n}" class="citation-chip" target="_self">[{n}]</a>' for n in nums]
        return " ".join(chips)

    # Match bracketed citations containing digits, commas, spaces, dashes
    return re.sub(r'(?:\*\*)?\[([\d\s,–-]+)\](?:\*\*)?', replace_citation, text)

sample = "RAG was introduced in [3]. Other works include [1, 2, 6] and **[4, 5]** and [7-9]. Non-citation: [Section 2], [a, b]."
print(format_citations(sample))
