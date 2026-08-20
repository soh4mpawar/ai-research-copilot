import time
import json
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    
    print("1. Loading App...")
    page.goto("http://localhost:8501", wait_until="load")
    time.sleep(5)
    
    print("2. Ensuring Dark Mode...")
    page.evaluate("""() => {
        const inputs = Array.from(document.querySelectorAll('input[type="checkbox"]'));
        const darkToggle = inputs.find(i => i.getAttribute('aria-label') === 'Dark Mode');
        if (darkToggle && !darkToggle.checked) {
            darkToggle.click();
        }
    }""")
    time.sleep(3)
    
    print("3. Running QA query to generate full answer & bibliography...")
    btn = page.locator("button:has-text('Run Research Pipeline')")
    if btn.count() > 0:
        btn.first.click()
        time.sleep(20)
        
    print("4. Auditing all inline tags, code, badges, and chips in QA view...")
    badge_audit = page.evaluate("""() => {
        const elements = Array.from(document.querySelectorAll('code, .badge-pill, .badge-outline, .badge-slate, .badge-strong, .badge-moderate, a.citation-chip, .source-meta code, [data-testid="stMarkdownContainer"] code'));
        return elements.map(el => {
            const cs = window.getComputedStyle(el);
            return {
                tagName: el.tagName,
                className: el.className,
                innerText: el.innerText.trim(),
                bg: cs.backgroundColor,
                color: cs.color,
                border: cs.border,
                borderColor: cs.borderColor
            };
        });
    }""")
    
    print(f"Found {len(badge_audit)} badge/code elements in QA view:")
    for b in badge_audit:
        print(f"[{b['tagName']}.{b['className']}] '{b['innerText']}' -> BG: {b['bg']}, Text: {b['color']}, Border: {b['borderColor']}")

    with open("scratch/badge_audit_results.json", "w", encoding="utf-8") as f:
        json.dump(badge_audit, f, indent=2)
        
    browser.close()
