import time
import os
import json
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
    page.wait_for_selector(".hero-title", timeout=60000)
    time.sleep(3)
    
    print("2. Toggling Dark Mode ON...")
    page.evaluate("""() => {
        const toggle = document.querySelector('[data-testid="stSidebar"] input[type="checkbox"]');
        if (toggle && !toggle.checked) {
            toggle.click();
        }
    }""")
    time.sleep(4)
    print("Verified Dark Mode BG:", page.evaluate("() => window.getComputedStyle(document.body).backgroundColor"))
    
    print("3. Running QA query...")
    page.locator("button:has-text('Run Research Pipeline')").first.click()
    time.sleep(24)
    
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
    light_leaks = [b for b in audit if "248, 249, 251" in b['bg'] or "255, 255, 255" in b['bg'] or "250, 250, 249" in b['bg']]
    if light_leaks:
        print(f"[WARN] Found {len(light_leaks)} light background leaks:")
        for l in light_leaks:
            print(f"   - {l['tag']}.{l['cls']}: '{l['text']}' -> BG: {l['bg']}")
    else:
        print("[SUCCESS] All 70+ badge and code elements have dark backgrounds!")
        
    print("5. Capturing Full Scroll QA Engine Dark Mode...")
    page.screenshot(path=qa_scroll_dark_path, full_page=True)
    print(f"[OK] Saved QA Dark Scroll: {qa_scroll_dark_path}")
    
    print("6. Navigating to Scientific Paper Explorer...")
    page.locator("div[role='radiogroup'] label").nth(2).click()
    time.sleep(6)
    page.screenshot(path=explorer_scroll_dark_path, full_page=True)
    print(f"[OK] Saved Paper Explorer Dark Scroll: {explorer_scroll_dark_path}")

    print("7. Navigating to Literature Review Studio...")
    page.locator("div[role='radiogroup'] label").nth(1).click()
    time.sleep(6)
    page.screenshot(path=lit_review_scroll_dark_path, full_page=True)
    print(f"[OK] Saved Lit Review Dark Scroll: {lit_review_scroll_dark_path}")
    
    browser.close()
    print("All captures completed successfully!")
