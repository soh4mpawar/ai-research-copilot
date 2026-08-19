import time
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    page.goto("http://localhost:8501", wait_until="load")
    time.sleep(4)
    
    print("Clicking Dark Mode label with force=True...")
    page.locator("label:has-text('Dark Mode')").first.click(force=True)
    time.sleep(3)
    
    page.screenshot(path="scratch/dark_mode_forced.png", full_page=True)
    print("Saved scratch/dark_mode_forced.png!")
    browser.close()
