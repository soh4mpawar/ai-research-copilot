import time
import os
from playwright.sync_api import sync_playwright

artifact_dir = r"C:\Users\soham\.gemini\antigravity\brain\5a470592-edee-4883-92fe-3f9d078e5ffb"
focused_after_path = os.path.join(artifact_dir, "citation_graph_focused_after.png")
focused_before_path = os.path.join(artifact_dir, "citation_graph_focused_before.png")

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
    
    # 4. Capture FOCUSED AFTER (Connected Subgraph, Hub Sizing, Styled Edges, Legend)
    print("4. Capturing Focused AFTER (Connected Subgraph)...")
    # Take screenshot of the main graph section
    page.screenshot(path=focused_after_path, full_page=False)
    print(f"[OK] Saved Focused AFTER: {focused_after_path}")
    
    # 5. Change Min Connections to "All Papers (0+)" and uncheck connected filter to show the BEFORE scatter
    print("5. Switching to All Papers (0+) for BEFORE comparison...")
    page.evaluate("""() => {
        const filterCb = Array.from(document.querySelectorAll('input[type="checkbox"]')).find(i => i.nextElementSibling && i.nextElementSibling.innerText.includes('Filter to Connected'));
        if (filterCb && filterCb.checked) {
            filterCb.click();
        }
    }""")
    time.sleep(6)
    
    page.screenshot(path=focused_before_path, full_page=False)
    print(f"[OK] Saved Focused BEFORE: {focused_before_path}")
    
    browser.close()
    print("Focused comparison capture complete!")
