import time
import os
from playwright.sync_api import sync_playwright

artifact_dir = r"C:\Users\soham\.gemini\antigravity\brain\5a470592-edee-4883-92fe-3f9d078e5ffb"
qa_fixed_path = os.path.join(artifact_dir, "qa_engine_citation_fix.png")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    
    print("1. Loading App and waiting for singleton orchestrator pre-warming...")
    page.goto("http://localhost:8501", wait_until="load")
    # Wait for the page title to appear (ensuring orchestrator is initialized)
    page.wait_for_selector(".academic-title", timeout=60000)
    time.sleep(3)
    
    print("2. Ensuring Dark Mode is Active...")
    page.evaluate("""() => {
        const inputs = Array.from(document.querySelectorAll('input[type="checkbox"]'));
        const darkToggle = inputs.find(i => i.getAttribute('aria-label') === 'Dark Mode');
        if (darkToggle && !darkToggle.checked) {
            darkToggle.click();
        }
    }""")
    time.sleep(3)
    
    print("3. Executing QA Query and waiting for full synthesis...")
    btn = page.locator("button:has-text('Run Research Pipeline')")
    if btn.count() > 0:
        btn.first.click()
        # Wait for the grounded synthesis header and citation chips
        page.wait_for_selector(".badge-strong:has-text('Evidence: Strong')", timeout=45000)
        time.sleep(3)
        
    print("4. Inspecting rendered citation chips...")
    chips = page.evaluate("""() => {
        const chipElements = Array.from(document.querySelectorAll('a.citation-chip'));
        return chipElements.map(c => ({
            text: c.innerText,
            href: c.getAttribute('href')
        }));
    }""")
    print(f"Rendered citation chips count: {len(chips)}")
    for c in chips:
        print(f"  Chip: {c['text']} -> {c['href']}")
        
    print("5. Capturing screenshot of QA Engine with consistent citation chips...")
    page.screenshot(path=qa_fixed_path, full_page=True)
    print(f"[OK] Saved screenshot to {qa_fixed_path}")
    
    browser.close()
    print("Verification complete!")
