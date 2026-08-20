import time
import os
from playwright.sync_api import sync_playwright

artifact_dir = r"C:\Users\soham\.gemini\antigravity\brain\5a470592-edee-4883-92fe-3f9d078e5ffb"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    
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
    page.locator("input[aria-label='Enter scientific research question:']").fill("What is RAG?")
    page.locator("button:has-text('Run Research Pipeline')").first.click()
    time.sleep(22)
    
    print("4. Switching to Retrieval Pipeline Transparency tab...")
    page.locator("div[role='tab']:has-text('Retrieval Pipeline Transparency')").first.click()
    time.sleep(3)
    
    # Scroll to dataframe
    page.locator("[data-testid='stDataFrame']").first.scroll_into_view_if_needed()
    time.sleep(2)
    
    # Test raw dataframe screenshot
    raw_shot = os.path.join(artifact_dir, "test_raw_df.png")
    page.screenshot(path=raw_shot, full_page=False)
    print(f"Saved {raw_shot}")
    
    # Test CSS Invert on canvas / dataframe
    page.evaluate("""() => {
        const dfs = document.querySelectorAll('[data-testid="stDataFrame"]');
        dfs.forEach(df => {
            df.style.filter = 'invert(0.9) hue-rotate(180deg) contrast(1.1)';
            df.style.borderRadius = '6px';
        });
    }""")
    time.sleep(1)
    inverted_shot = os.path.join(artifact_dir, "test_inverted_df.png")
    page.screenshot(path=inverted_shot, full_page=False)
    print(f"Saved {inverted_shot}")
    
    browser.close()
