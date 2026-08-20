import time
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 720})
    page.goto("http://localhost:8502", wait_until="load")
    time.sleep(2)
    
    # Click selectbox container to open dropdown
    page.locator("[data-testid='stSelectbox']").first.click()
    time.sleep(1)
    
    # Type RRF
    page.keyboard.type("RRF")
    time.sleep(1)
    
    # Click option
    option_locator = page.locator("[role='option']").first
    print("Top option text:", option_locator.inner_text())
    option_locator.click()
    time.sleep(2)
    
    # Check page state
    body_text = page.locator("body").inner_text()
    print("Body text snippet:\n", body_text[:400])
    
    browser.close()
