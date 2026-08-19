import time
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    page.goto("http://localhost:8501", wait_until="load")
    time.sleep(4)
    
    print("Checking Dark Mode checkbox...")
    # Click the toggle switch directly
    toggle_input = page.locator("input[aria-label='Dark Mode']")
    toggle_input.click()
    time.sleep(3)
    
    # Run Research Pipeline
    page.locator("button:has-text('Run Research Pipeline')").first.click()
    time.sleep(6)
    
    page.screenshot(path="scratch/dark_mode_active_qa.png", full_page=True)
    print("Saved dark_mode_active_qa.png!")
    browser.close()
