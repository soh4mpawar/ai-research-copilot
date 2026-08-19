import time
import os
from playwright.sync_api import sync_playwright

artifact_dir = r"C:\Users\soham\.gemini\antigravity\brain\5a470592-edee-4883-92fe-3f9d078e5ffb"
focused_before_path = os.path.join(artifact_dir, "citation_graph_before_scatter.png")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    
    print("1. Loading App...")
    page.goto("http://localhost:8501", wait_until="load")
    page.wait_for_selector(".hero-title", timeout=60000)
    time.sleep(3)
    
    print("2. Navigating to Citation Graph Page...")
    page.locator("div[role='radiogroup'] label").nth(3).click()
    time.sleep(4)
    
    print("3. Enabling Dark Mode...")
    page.evaluate("""() => {
        const inputs = Array.from(document.querySelectorAll('input[type="checkbox"]'));
        const darkToggle = inputs.find(i => i.getAttribute('aria-label') === 'Dark Mode');
        if (darkToggle && !darkToggle.checked) {
            darkToggle.click();
        }
    }""")
    time.sleep(3)
    
    print("4. Unchecking connected filter on the main page...")
    page.locator("section[data-testid='stMain'] [data-testid='stCheckbox'] label").click()
    time.sleep(6)
    
    page.screenshot(path=focused_before_path, full_page=False)
    print(f"[OK] Saved BEFORE: {focused_before_path}")
    
    browser.close()
