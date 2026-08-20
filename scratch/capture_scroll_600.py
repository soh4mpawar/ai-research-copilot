import time
import os
from playwright.sync_api import sync_playwright

artifact_dir = r"C:\Users\soham\.gemini\antigravity\brain\5a470592-edee-4883-92fe-3f9d078e5ffb"
matrix_table_dark_path = os.path.join(artifact_dir, "comparative_matrix_table_dark.png")
matrix_table_light_path = os.path.join(artifact_dir, "comparative_matrix_table_light.png")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    
    print("1. Loading App...")
    page.goto("http://localhost:8501", wait_until="load")
    page.wait_for_selector(".academic-title", timeout=60000)
    time.sleep(3)
    
    print("2. Toggling Dark Mode ON...")
    page.evaluate("""() => {
        const toggle = document.querySelector('[data-testid="stSidebar"] input[type="checkbox"]');
        if (toggle && !toggle.checked) {
            toggle.click();
        }
    }""")
    time.sleep(3)
    
    print("3. Clicking Run Research Pipeline...")
    page.locator("button:has-text('Run Research Pipeline')").first.click()
    
    # Wait for the spinner to disappear and answer to be visible
    print("4. Waiting for research pipeline execution (25s)...")
    time.sleep(25)
    
    print("5. Switching to Retrieval Pipeline Transparency tab...")
    page.locator("div[role='tab']:has-text('Retrieval Pipeline Transparency')").first.click()
    time.sleep(3)
    
    print("6. Scrolling down to Comparative Rank Matrix table...")
    page.evaluate("() => window.scrollTo(0, 600)")
    time.sleep(2)
    
    page.screenshot(path=matrix_table_dark_path, full_page=False)
    print(f"[OK] Saved Comparative Rank Matrix Table Dark: {matrix_table_dark_path}")
    
    # 7. Light mode test for QA table
    print("7. Toggling Light Mode...")
    page.evaluate("""() => {
        const toggle = document.querySelector('[data-testid="stSidebar"] input[type="checkbox"]');
        if (toggle && toggle.checked) {
            toggle.click();
        }
    }""")
    time.sleep(3)
    page.locator("div[role='tab']:has-text('Retrieval Pipeline Transparency')").first.click()
    time.sleep(2)
    page.evaluate("() => window.scrollTo(0, 600)")
    time.sleep(2)
    page.screenshot(path=matrix_table_light_path, full_page=False)
    print(f"[OK] Saved Comparative Rank Matrix Table Light: {matrix_table_light_path}")
    
    browser.close()
    print("Captures complete!")
