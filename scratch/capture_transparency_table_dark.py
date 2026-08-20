import time
import os
from playwright.sync_api import sync_playwright

artifact_dir = r"C:\Users\soham\.gemini\antigravity\brain\5a470592-edee-4883-92fe-3f9d078e5ffb"
matrix_table_dark_path = os.path.join(artifact_dir, "comparative_matrix_table_dark.png")
eval_table_dark_path = os.path.join(artifact_dir, "eval_benchmark_table_dark.png")
lit_review_table_dark_path = os.path.join(artifact_dir, "lit_review_matrix_dark.png")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    
    print("1. Loading App...")
    page.goto("http://localhost:8501", wait_until="load")
    time.sleep(3)
    
    print("2. Toggling Dark Mode ON...")
    page.evaluate("""() => {
        const toggle = document.querySelector('[data-testid="stSidebar"] input[type="checkbox"]');
        if (toggle && !toggle.checked) {
            toggle.click();
        }
    }""")
    time.sleep(3)
    
    print("3. Executing QA query...")
    page.locator("input[aria-label='Enter scientific research question:']").fill("What is RAG and why was it introduced?")
    page.locator("button:has-text('Run Research Pipeline')").first.click()
    time.sleep(22)
    
    print("4. Switching to Retrieval Pipeline Transparency tab...")
    page.locator("div[role='tab']:has-text('Retrieval Pipeline Transparency')").first.click()
    time.sleep(4)
    
    print("5. Scrolling to Comparative Rank Matrix table...")
    df_locator = page.locator("[data-testid='stDataFrame']").first
    df_locator.scroll_into_view_if_needed()
    time.sleep(2)
    
    page.screenshot(path=matrix_table_dark_path, full_page=False)
    print(f"[OK] Saved Comparative Rank Matrix Table Dark: {matrix_table_dark_path}")
    
    # 6. Evaluation Dashboard Table
    print("6. Navigating to Evaluation & RAGAS Dashboard...")
    page.locator("div[role='radiogroup'] label").nth(4).click()
    time.sleep(5)
    page.locator("[data-testid='stDataFrame']").first.scroll_into_view_if_needed()
    time.sleep(2)
    page.screenshot(path=eval_table_dark_path, full_page=False)
    print(f"[OK] Saved Eval Benchmark Table Dark: {eval_table_dark_path}")

    # 7. Literature Review Matrix Table
    print("7. Navigating to Literature Review Studio...")
    page.locator("div[role='radiogroup'] label").nth(1).click()
    time.sleep(5)
    page.locator("button:has-text('Synthesize Literature Review')").first.click()
    time.sleep(20)
    page.locator("[data-testid='stDataFrame']").first.scroll_into_view_if_needed()
    time.sleep(2)
    page.screenshot(path=lit_review_table_dark_path, full_page=False)
    print(f"[OK] Saved Literature Review Matrix Dark: {lit_review_table_dark_path}")
    
    browser.close()
    print("All table captures completed successfully!")
