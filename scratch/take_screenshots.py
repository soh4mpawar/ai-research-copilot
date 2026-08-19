import time
from playwright.sync_api import sync_playwright

artifact_dir = r"C:\Users\soham\.gemini\antigravity\brain\5a470592-edee-4883-92fe-3f9d078e5ffb"
qa_img = f"{artifact_dir}\\qa_engine_academic_white.png"
eval_img = f"{artifact_dir}\\eval_dashboard_academic_white.png"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    
    print("1. Loading QA Page...")
    page.goto("http://localhost:8501", wait_until="networkidle")
    time.sleep(3)
    
    # Run Research Pipeline
    try:
        page.locator("button:has-text('Run Research Pipeline')").first.click()
        page.wait_for_selector(".academic-title:has-text('Scientific Research Synthesis')", timeout=35000)
        time.sleep(3)
    except Exception as e:
        print("Note on QA answer:", e)
        time.sleep(4)
        
    page.screenshot(path=qa_img, full_page=True)
    print("Saved QA Screenshot:", qa_img)
    
    # Switch to Evaluation Dashboard
    print("2. Switching to Evaluation & RAGAS Dashboard...")
    try:
        page.locator("div[role='radiogroup'] label").nth(4).click()
        page.wait_for_selector("div.academic-title:has-text('RAGAS Evaluation')", timeout=30000)
        # Wait for Plotly charts to render
        page.wait_for_selector(".js-plotly-plot", timeout=15000)
        time.sleep(4)
        page.screenshot(path=eval_img, full_page=True)
        print("Saved Eval Dashboard Screenshot:", eval_img)
    except Exception as e:
        print("Error capturing Eval Dashboard:", e)
        
    browser.close()
    print("Capture complete.")
