import time
import os
from playwright.sync_api import sync_playwright

artifact_dir = r"C:\Users\soham\.gemini\antigravity\brain\5a470592-edee-4883-92fe-3f9d078e5ffb"
qa_path = os.path.join(artifact_dir, "qa_engine_structural_pass.png")
explorer_path = os.path.join(artifact_dir, "paper_explorer_structural_pass.png")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1050})
    
    print("1. Loading QA Page...")
    page.goto("http://localhost:8501", wait_until="load")
    time.sleep(4)
    
    # Run query
    try:
        btn = page.locator("button:has-text('Run Research Pipeline')")
        if btn.count() > 0:
            btn.first.click()
            time.sleep(6)
    except Exception as e:
        print("Note on button:", e)
        
    page.screenshot(path=qa_path, full_page=True)
    print("[OK] Saved QA Screenshot:", qa_path)
    
    # Switch to Scientific Paper Explorer
    print("2. Switching to Scientific Paper Explorer...")
    try:
        page.locator("div[role='radiogroup'] label").nth(2).click()
        time.sleep(4)
        page.screenshot(path=explorer_path, full_page=True)
        print("[OK] Saved Explorer Screenshot:", explorer_path)
    except Exception as e:
        print("Error capturing Paper Explorer:", e)
        
    browser.close()
    print("Done!")
