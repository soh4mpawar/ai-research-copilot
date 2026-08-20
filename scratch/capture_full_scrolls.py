import time
import os
from playwright.sync_api import sync_playwright

artifact_dir = r"C:\Users\soham\.gemini\antigravity\brain\5a470592-edee-4883-92fe-3f9d078e5ffb"
qa_scroll_dark_path = os.path.join(artifact_dir, "qa_engine_scroll_dark.png")
explorer_scroll_dark_path = os.path.join(artifact_dir, "paper_explorer_scroll_dark.png")
lit_review_scroll_dark_path = os.path.join(artifact_dir, "lit_review_scroll_dark.png")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    
    print("1. Loading App...")
    page.goto("http://localhost:8501", wait_until="load")
    page.wait_for_selector(".hero-title", timeout=60000)
    time.sleep(3)
    
    # Check if dark mode is active by checking body bg
    body_bg = page.evaluate("() => window.getComputedStyle(document.body).backgroundColor")
    print(f"Current Body BG: {body_bg}")
    
    if "22" not in body_bg and "26" not in body_bg:
        print("Toggling Dark Mode ON...")
        page.locator("label:has-text('Dark Mode')").first.click()
        time.sleep(4)
        
    print(f"Verified Dark Mode BG: {page.evaluate('() => window.getComputedStyle(document.body).backgroundColor')}")
    
    print("2. Running QA query...")
    page.locator("button:has-text('Run Research Pipeline')").first.click()
    time.sleep(22)
    
    print("3. Capturing Full Scroll QA Engine Dark Mode...")
    page.screenshot(path=qa_scroll_dark_path, full_page=True)
    print(f"[OK] Saved QA Dark Scroll: {qa_scroll_dark_path}")
    
    print("4. Navigating to Scientific Paper Explorer...")
    page.locator("div[role='radiogroup'] label").nth(2).click()
    time.sleep(5)
    page.screenshot(path=explorer_scroll_dark_path, full_page=True)
    print(f"[OK] Saved Paper Explorer Dark Scroll: {explorer_scroll_dark_path}")

    print("5. Navigating to Literature Review Studio...")
    page.locator("div[role='radiogroup'] label").nth(1).click()
    time.sleep(5)
    page.screenshot(path=lit_review_scroll_dark_path, full_page=True)
    print(f"[OK] Saved Lit Review Dark Scroll: {lit_review_scroll_dark_path}")
    
    browser.close()
    print("Captures complete!")
