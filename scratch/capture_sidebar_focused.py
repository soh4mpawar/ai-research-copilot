import time
import os
from playwright.sync_api import sync_playwright

artifact_dir = r"C:\Users\soham\.gemini\antigravity\brain\5a470592-edee-4883-92fe-3f9d078e5ffb"
sidebar_light_path = os.path.join(artifact_dir, "sidebar_light_mode.png")
sidebar_dark_path = os.path.join(artifact_dir, "sidebar_dark_mode.png")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    
    print("1. Loading App...")
    page.goto("http://localhost:8501", wait_until="load")
    page.wait_for_selector(".academic-title", timeout=60000)
    time.sleep(4)
    
    # 1. Capture Sidebar in Light Mode
    print("2. Capturing Sidebar in Light Mode...")
    sidebar_elem = page.locator('[data-testid="stSidebar"]')
    sidebar_elem.screenshot(path=sidebar_light_path)
    print(f"[OK] Saved Light Sidebar Screenshot: {sidebar_light_path}")
    
    # 2. Toggle Dark Mode
    print("3. Toggling Dark Mode...")
    page.evaluate("""() => {
        const inputs = Array.from(document.querySelectorAll('input[type="checkbox"]'));
        const darkToggle = inputs.find(i => i.getAttribute('aria-label') === 'Dark Mode');
        if (darkToggle && !darkToggle.checked) {
            darkToggle.click();
        }
    }""")
    time.sleep(3)
    
    # 3. Capture Sidebar in Dark Mode
    print("4. Capturing Sidebar in Dark Mode...")
    sidebar_elem = page.locator('[data-testid="stSidebar"]')
    sidebar_elem.screenshot(path=sidebar_dark_path)
    print(f"[OK] Saved Dark Sidebar Screenshot: {sidebar_dark_path}")
    
    browser.close()
    print("Sidebar captures completed successfully!")
