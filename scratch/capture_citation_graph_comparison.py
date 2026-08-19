import time
import os
from playwright.sync_api import sync_playwright

artifact_dir = r"C:\Users\soham\.gemini\antigravity\brain\5a470592-edee-4883-92fe-3f9d078e5ffb"
after_dark_path = os.path.join(artifact_dir, "citation_graph_after_dark.png")
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
    
    # 3. Capture Graph Area in Light Mode (Filtered Connected Subgraph)
    print("3. Capturing Light Mode Citation Graph...")
    page.screenshot(path=after_light_path, full_page=True)
    print(f"[OK] Saved Light Graph: {after_light_path}")
    
    # 4. Toggle Dark Mode
    print("4. Toggling Dark Mode...")
    page.evaluate("""() => {
        const inputs = Array.from(document.querySelectorAll('input[type="checkbox"]'));
        const darkToggle = inputs.find(i => i.getAttribute('aria-label') === 'Dark Mode');
        if (darkToggle && !darkToggle.checked) {
            darkToggle.click();
        }
    }""")
    time.sleep(4)
    
    # 5. Capture Graph Area in Dark Mode
    print("5. Capturing Dark Mode Citation Graph...")
    page.screenshot(path=after_dark_path, full_page=True)
    print(f"[OK] Saved Dark Graph: {after_dark_path}")
    
    browser.close()
    print("Graph comparison captures complete!")
