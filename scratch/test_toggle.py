import time
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    page.goto("http://localhost:8501", wait_until="load")
    time.sleep(4)
    
    print("Clicking Dark Mode label...")
    page.locator("[data-testid='stSidebar'] label:has-text('Dark Mode')").click()
    time.sleep(4)
    page.screenshot(path="scratch/toggle_test_dark.png", full_page=True)
    print("Screenshot saved to scratch/toggle_test_dark.png!")
    browser.close()
