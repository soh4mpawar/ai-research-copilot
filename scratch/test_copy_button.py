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
        viewport={"width": 1440, "height": 1080}
    )
    page = context.new_page()
    
    print("1. Loading App...")
    page.goto("http://localhost:8501", wait_until="load")
    page.wait_for_selector(".hero-title", timeout=60000)
    time.sleep(3)
    
    print("2. Toggling Dark Mode ON...")
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
    
    print("4. Scrolling to Action Strip...")
    page.evaluate("() => window.scrollTo(0, 750)")
    time.sleep(2)
    
    # Capture QA Action Strip in Dark Mode
    page.screenshot(path=qa_action_bar_dark_path, full_page=False)
    print(f"[OK] Saved QA Dark Action Strip: {qa_action_bar_dark_path}")
    
    # Test clicking the Copy Button inside the iframe
    print("5. Clicking Copy Button inside iframe...")
    iframe = page.frame_locator("iframe").first
    copy_btn = iframe.locator("#copyBtn")
    if copy_btn.count() > 0:
        copy_btn.click()
        time.sleep(0.5)
        label_text = iframe.locator("#copyLabel").inner_text()
        print(f"Post-click label: {label_text}")
        
    # Toggle Light Mode to test Light Mode styling
    print("6. Toggling Light Mode...")
    page.evaluate("""() => {
        const toggle = document.querySelector('[data-testid="stSidebar"] input[type="checkbox"]');
        if (toggle && toggle.checked) {
            toggle.click();
        }
    }""")
    time.sleep(3)
    page.evaluate("() => window.scrollTo(0, 750)")
    time.sleep(2)
    page.screenshot(path=qa_action_bar_light_path, full_page=False)
    print(f"[OK] Saved QA Light Action Strip: {qa_action_bar_light_path}")
    
    browser.close()
    print("Copy button tests and captures complete!")
