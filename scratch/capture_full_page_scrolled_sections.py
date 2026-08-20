import time
import os
from playwright.sync_api import sync_playwright

artifact_dir = r"C:\Users\soham\.gemini\antigravity\brain\5a470592-edee-4883-92fe-3f9d078e5ffb"
qa_bottom_dark_path = os.path.join(artifact_dir, "qa_engine_bottom_dark.png")
explorer_full_path = os.path.join(artifact_dir, "paper_explorer_full_dark.png")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1800})
    
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
    time.sleep(3)
    
    print("3. Running QA query...")
    page.locator("button:has-text('Run Research Pipeline')").first.click()
    time.sleep(22)
    
    # Scroll down to Grounded Source Bibliography
    print("4. Scrolling to bibliography section...")
    page.evaluate("() => window.scrollTo(0, 1100)")
    time.sleep(2)
    page.screenshot(path=qa_bottom_dark_path, full_page=False)
    print(f"[OK] Saved QA Bottom Dark: {qa_bottom_dark_path}")
    
    # Navigate to Paper Explorer and capture full page
    print("5. Navigating to Scientific Paper Explorer...")
    page.locator("div[role='radiogroup'] label").nth(2).click()
    time.sleep(6)
    page.screenshot(path=explorer_full_path, full_page=False)
    print(f"[OK] Saved Explorer Full: {explorer_full_path}")
    
    browser.close()
