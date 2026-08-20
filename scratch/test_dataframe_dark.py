import time
import os
from playwright.sync_api import sync_playwright

artifact_dir = r"C:\Users\soham\.gemini\antigravity\brain\5a470592-edee-4883-92fe-3f9d078e5ffb"
df_screenshot_path = os.path.join(artifact_dir, "dataframe_matrix_dark.png")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    
    print("1. Loading App...")
    page.goto("http://localhost:8501", wait_until="load")
    time.sleep(3)
    
    print("2. Toggling Dark Mode...")
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
    
    print("4. Switching to Retrieval Pipeline Transparency tab...")
    # Click tab 2
    tab_btn = page.locator("[role='tab']").nth(1)
    tab_btn.click()
    time.sleep(4)
    
    print("5. Scrolling to Comparative Rank Matrix table...")
    page.locator("[data-testid='stDataFrame']").first.scroll_into_view_if_needed()
    time.sleep(2)
    
    page.screenshot(path=df_screenshot_path, full_page=False)
    print(f"[OK] Saved DataFrame Matrix Screenshot: {df_screenshot_path}")
    
    browser.close()
