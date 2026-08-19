import time
import os
from playwright.sync_api import sync_playwright

artifact_dir = r"C:\Users\soham\.gemini\antigravity\brain\5a470592-edee-4883-92fe-3f9d078e5ffb"
before_path = os.path.join(artifact_dir, "citation_graph_before.png")
after_dark_path = os.path.join(artifact_dir, "citation_graph_after.png")
after_light_path = os.path.join(artifact_dir, "citation_graph_after_light.png")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    
    print("1. Loading App...")
    page.goto("http://localhost:8501", wait_until="load")
    page.wait_for_selector(".hero-title", timeout=60000)
    time.sleep(3)
    
    # 2. Switch to Citation Graph Page
    print("2. Navigating to Citation Graph Page...")
    page.locator("div[role='radiogroup'] label").nth(3).click()
    time.sleep(5)
    
    # 3. Ensure Dark Mode
    print("3. Enabling Dark Mode...")
    page.evaluate("""() => {
        const inputs = Array.from(document.querySelectorAll('input[type="checkbox"]'));
        const darkToggle = inputs.find(i => i.getAttribute('aria-label') === 'Dark Mode');
        if (darkToggle && !darkToggle.checked) {
            darkToggle.click();
        }
    }""")
    time.sleep(4)
    
    # 4. Capture "AFTER" (Connected Subgraph, Hub Sizing, Legend, Dark Card)
    print("4. Capturing AFTER (Connected Subgraph)...")
    page.screenshot(path=after_dark_path, full_page=True)
    print(f"[OK] Saved AFTER: {after_dark_path}")
    
    # 5. Capture "BEFORE" (Unchecking Connected filter to show all 200 nodes scatter)
    print("5. Capturing BEFORE (Full 200-node scattered network)...")
    filter_checkbox = page.locator("label:has-text('Filter to Connected Citation Network')")
    if filter_checkbox.count() > 0:
        filter_checkbox.click()
        time.sleep(5)
    page.screenshot(path=before_path, full_page=True)
    print(f"[OK] Saved BEFORE: {before_path}")
    
    # 6. Re-check connected filter & toggle Light Mode for Light version
    print("6. Capturing Light Mode AFTER...")
    if filter_checkbox.count() > 0:
        filter_checkbox.click()
        time.sleep(3)
        
    page.evaluate("""() => {
        const inputs = Array.from(document.querySelectorAll('input[type="checkbox"]'));
        const darkToggle = inputs.find(i => i.getAttribute('aria-label') === 'Dark Mode');
        if (darkToggle && darkToggle.checked) {
            darkToggle.click();
        }
    }""")
    time.sleep(4)
    page.screenshot(path=after_light_path, full_page=True)
    print(f"[OK] Saved Light AFTER: {after_light_path}")
    
    browser.close()
    print("All captures completed successfully!")
