import time
import os
from playwright.sync_api import sync_playwright

artifact_dir = r"C:\Users\soham\.gemini\antigravity\brain\5a470592-edee-4883-92fe-3f9d078e5ffb"
qa_action_bar_dark_path = os.path.join(artifact_dir, "copy_button_qa_dark.png")
qa_action_bar_light_path = os.path.join(artifact_dir, "copy_button_qa_light.png")

with sync_playwright() as p:
    browser = p.chromium.launch()
    context = browser.new_context(
        permissions=["clipboard-read", "clipboard-write"],
        viewport={"width": 1440, "height": 950}
    )
    page = context.new_page()
    
    print("1. Loading App...")
    page.goto("http://localhost:8501", wait_until="load")
    time.sleep(3)
    
    print("2. Running QA query...")
    page.locator("button:has-text('Run Research Pipeline')").first.click()
    time.sleep(22)
    
    # Scroll to action strip in Light Mode
    print("3. Capturing Light Mode Action Strip...")
    page.evaluate("() => { const el = document.querySelector('iframe'); if(el) el.scrollIntoView({block: 'center'}); }")
    time.sleep(2)
    page.screenshot(path=qa_action_bar_light_path, full_page=False)
    print(f"[OK] Saved QA Light: {qa_action_bar_light_path}")
    
    # Toggle Dark Mode
    print("4. Toggling Dark Mode...")
    page.evaluate("""() => {
        const toggle = document.querySelector('[data-testid="stSidebar"] input[type="checkbox"]');
        if (toggle && !toggle.checked) {
            toggle.click();
        }
    }""")
    time.sleep(3)
    page.evaluate("() => { const el = document.querySelector('iframe'); if(el) el.scrollIntoView({block: 'center'}); }")
    time.sleep(2)
    page.screenshot(path=qa_action_bar_dark_path, full_page=False)
    print(f"[OK] Saved QA Dark: {qa_action_bar_dark_path}")
    
    browser.close()
