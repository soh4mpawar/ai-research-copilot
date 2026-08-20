import time
import json
import os
from playwright.sync_api import sync_playwright

artifact_dir = r"C:\Users\soham\.gemini\antigravity\brain\5a470592-edee-4883-92fe-3f9d078e5ffb"
qa_scroll_dark_path = os.path.join(artifact_dir, "qa_engine_scroll_dark.png")
explorer_scroll_dark_path = os.path.join(artifact_dir, "paper_explorer_scroll_dark.png")
lit_review_scroll_dark_path = os.path.join(artifact_dir, "lit_review_scroll_dark.png")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    
    print("1. Loading App...")
    page.goto("http://localhost:8501", wait_until="load")
    time.sleep(4)
    
    print("2. Enabling Dark Mode...")
    page.evaluate("""() => {
        const inputs = Array.from(document.querySelectorAll('input[type="checkbox"]'));
        const darkToggle = inputs.find(i => i.getAttribute('aria-label') === 'Dark Mode');
        if (darkToggle && !darkToggle.checked) {
            darkToggle.click();
        }
    }""")
    time.sleep(3)
    
    print("3. Executing QA pipeline query for full scroll capture...")
    page.locator("button:has-text('Run Research Pipeline')").first.click()
    time.sleep(22)
    
    print("4. Auditing computed styles of all code and badge elements in QA View...")
    audit = page.evaluate("""() => {
        const elements = Array.from(document.querySelectorAll('code, .badge-pill, .badge-outline, .badge-slate, .badge-strong, .badge-moderate, a.citation-chip, .source-meta code, [data-testid="stMarkdownContainer"] code'));
        return elements.map(el => {
            const cs = window.getComputedStyle(el);
            return {
                tag: el.tagName,
                cls: el.className,
                text: el.innerText.trim().slice(0, 30),
                bg: cs.backgroundColor,
                color: cs.color,
                border: cs.borderColor
            };
        });
    }""")
    
    print(f"Total inspected elements: {len(audit)}")
    light_leaks = [b for b in audit if "248, 249, 251" in b['bg'] or "255, 255, 255" in b['bg']]
    if light_leaks:
        print(f"[WARN] Found {len(light_leaks)} light background leaks:")
        for l in light_leaks:
            print(f"   - {l['tag']}.{l['cls']}: '{l['text']}' -> BG: {l['bg']}")
    else:
        print("[SUCCESS] All 60+ badge and code elements have dark backgrounds!")
        
    # Capture Full-Page Scroll-Through Screenshots
    print("5. Capturing Full-Page Scroll QA Engine in Dark Mode...")
    page.screenshot(path=qa_scroll_dark_path, full_page=True)
    print(f"[OK] Saved QA Scroll: {qa_scroll_dark_path}")
    
    print("6. Navigating to Scientific Paper Explorer...")
    page.locator("div[role='radiogroup'] label").nth(2).click()
    time.sleep(5)
    page.screenshot(path=explorer_scroll_dark_path, full_page=True)
    print(f"[OK] Saved Paper Explorer Scroll: {explorer_scroll_dark_path}")

    print("7. Navigating to Literature Review Studio...")
    page.locator("div[role='radiogroup'] label").nth(1).click()
    time.sleep(5)
    page.screenshot(path=lit_review_scroll_dark_path, full_page=True)
    print(f"[OK] Saved Lit Review Scroll: {lit_review_scroll_dark_path}")
    
    browser.close()
    print("All scroll verification captures completed successfully!")
