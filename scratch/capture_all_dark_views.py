import time
import os
from playwright.sync_api import sync_playwright

artifact_dir = r"C:\Users\soham\.gemini\antigravity\brain\5a470592-edee-4883-92fe-3f9d078e5ffb"

views = [
    ("Research QA Engine", 0, "qa_engine_dark_mode.png"),
    ("Literature Review Studio", 1, "lit_review_dark_mode.png"),
    ("Scientific Paper Explorer", 2, "paper_explorer_dark_mode.png"),
    ("Citation Network Graph", 3, "citation_graph_dark_mode.png"),
    ("Evaluation & RAGAS Dashboard", 4, "eval_dashboard_dark_mode.png"),
]

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    
    print("1. Loading App...")
    page.goto("http://localhost:8501", wait_until="load")
    time.sleep(4)
    
    print("2. Ensuring Dark Mode is Active...")
    page.evaluate("""() => {
        const inputs = Array.from(document.querySelectorAll('input[type="checkbox"]'));
        const darkToggle = inputs.find(i => i.getAttribute('aria-label') === 'Dark Mode');
        if (darkToggle && !darkToggle.checked) {
            darkToggle.click();
        }
    }""")
    time.sleep(3)
    
    for name, idx, filename in views:
        print(f"Capturing {name} (index {idx})...")
        page.locator("div[role='radiogroup'] label").nth(idx).click()
        time.sleep(4)
        
        if idx == 0:
            try:
                page.locator("button:has-text('Run Research Pipeline')").first.click()
                time.sleep(8)
            except Exception as e:
                pass
                
        out_path = os.path.join(artifact_dir, filename)
        page.screenshot(path=out_path, full_page=True)
        print(f"[OK] Saved {filename}")
        
    browser.close()
    print("All 5 dark mode views captured successfully!")
