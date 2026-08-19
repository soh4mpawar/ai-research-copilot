import time
import os
from playwright.sync_api import sync_playwright

artifact_dir = r"C:\Users\soham\.gemini\antigravity\brain\5a470592-edee-4883-92fe-3f9d078e5ffb"
qa_dark_path = os.path.join(artifact_dir, "qa_engine_dark_mode.png")
eval_dark_path = os.path.join(artifact_dir, "eval_dashboard_dark_mode.png")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    
    print("1. Loading App...")
    page.goto("http://localhost:8501", wait_until="load")
    time.sleep(5)
    
    print("2. Toggling Dark Mode...")
    page.evaluate("""() => {
        const inputs = Array.from(document.querySelectorAll('input[type="checkbox"]'));
        const darkToggle = inputs.find(i => i.getAttribute('aria-label') === 'Dark Mode');
        if (darkToggle) {
            darkToggle.click();
        }
    }""")
    time.sleep(3)
    
    # Run Research Pipeline on QA view
    print("3. Running QA Query in Dark Mode...")
    try:
        btn = page.locator("button:has-text('Run Research Pipeline')")
        if btn.count() > 0:
            btn.first.click()
            time.sleep(9)
    except Exception as e:
        print("Note on QA query:", e)
        time.sleep(4)
        
    page.screenshot(path=qa_dark_path, full_page=True)
    print("[OK] Saved QA Dark Mode Screenshot:", qa_dark_path)
    
    # Switch to Evaluation Dashboard
    print("4. Switching to Evaluation Dashboard in Dark Mode...")
    try:
        page.locator("div[role='radiogroup'] label").nth(4).click()
        time.sleep(5)
        page.screenshot(path=eval_dark_path, full_page=True)
        print("[OK] Saved Eval Dark Mode Screenshot:", eval_dark_path)
    except Exception as e:
        print("Error capturing Eval Dashboard in Dark Mode:", e)
        
    browser.close()
    print("Final capture complete!")
