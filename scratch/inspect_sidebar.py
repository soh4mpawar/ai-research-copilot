import time
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    page.goto("http://localhost:8501", wait_until="load")
    time.sleep(5)
    
    # List all inputs and checkboxes
    inputs = page.locator("[data-testid='stSidebar'] input")
    print("Sidebar inputs count:", inputs.count())
    for i in range(inputs.count()):
        print(f"Input {i}: type={inputs.nth(i).get_attribute('type')}, aria-label={inputs.nth(i).get_attribute('aria-label')}")
        
    labels = page.locator("[data-testid='stSidebar'] label")
    print("Sidebar labels count:", labels.count())
    for i in range(labels.count()):
        print(f"Label {i}: text={labels.nth(i).inner_text()}")
        
    browser.close()
