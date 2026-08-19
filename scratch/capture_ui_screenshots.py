import time
import os
from playwright.sync_api import sync_playwright

def main():
    artifact_dir = r"C:\Users\soham\.gemini\antigravity\brain\5a470592-edee-4883-92fe-3f9d078e5ffb"
    os.makedirs(artifact_dir, exist_ok=True)
    
    qa_path = os.path.join(artifact_dir, "qa_engine_light_academic.png")
    eval_path = os.path.join(artifact_dir, "eval_dashboard_light_academic.png")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 960})
        page = context.new_page()
        
        print("Navigating to QA page...")
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=45000)
        time.sleep(4)
        
        # Click Run Research Pipeline button or wait for page elements
        try:
            btn = page.locator("button:has-text('Run Research Pipeline')")
            if btn.count() > 0:
                print("Clicking Run Research Pipeline button...")
                btn.first.click()
                time.sleep(8)
        except Exception as e:
            print("Note on button click:", e)
            
        page.screenshot(path=qa_path, full_page=True)
        print(f"Captured QA page to: {qa_path}")
        
        # Navigate to Eval Dashboard
        print("Navigating to Evaluation & RAGAS Dashboard...")
        try:
            # Click radio button for Eval Dashboard
            radio = page.locator("text='📊 Evaluation & RAGAS Dashboard'")
            if radio.count() > 0:
                radio.first.click()
                time.sleep(4)
        except Exception as e:
            print("Note on nav click:", e)
            
        page.screenshot(path=eval_path, full_page=True)
        print(f"Captured Eval dashboard to: {eval_path}")
        
        browser.close()

if __name__ == "__main__":
    main()
