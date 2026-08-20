import time
import os
from playwright.sync_api import sync_playwright

artifact_dir = r"C:\Users\soham\.gemini\antigravity\brain\5a470592-edee-4883-92fe-3f9d078e5ffb"
df_matrix_dark_path = os.path.join(artifact_dir, "dataframe_matrix_dark.png")
eval_df_dark_path = os.path.join(artifact_dir, "eval_benchmark_table_dark.png")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    
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
    
    print("3. Clicking Preset Query button...")
    page.locator("button:has-text('What is RAG and why was it introduced?')").first.click()
    
    print("4. Waiting for pipeline execution to complete...")
    # Wait for the tabs to appear
    page.wait_for_selector("[data-testid='stTabs']", timeout=60000)
    time.sleep(3)
    
    print("5. Switching to Retrieval Pipeline Transparency tab...")
    page.locator("[data-testid='stTabs'] [data-baseweb='tab']").nth(1).click()
    time.sleep(3)
    
    print("6. Scrolling to Comparative Rank Matrix table...")
    df_elem = page.locator("[data-testid='stDataFrame']").first
    df_elem.scroll_into_view_if_needed()
    time.sleep(2)
    
    page.screenshot(path=df_matrix_dark_path, full_page=False)
    print(f"[OK] Saved Comparative Rank Matrix Dark Screenshot: {df_matrix_dark_path}")
    
    # 7. Check Evaluation Dashboard Table
    print("7. Navigating to Evaluation & RAGAS Dashboard...")
    page.locator("div[role='radiogroup'] label").nth(4).click()
    time.sleep(5)
    page.locator("[data-testid='stDataFrame']").first.scroll_into_view_if_needed()
    time.sleep(2)
    page.screenshot(path=eval_df_dark_path, full_page=False)
    print(f"[OK] Saved Eval Benchmark Table Dark Screenshot: {eval_df_dark_path}")
    
    browser.close()
    print("All dataframe captures complete!")
