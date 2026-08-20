import time
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 720})
    page.goto("http://localhost:8502", wait_until="load")
    time.sleep(3)
    
    input_box = page.locator("[data-testid='stSelectbox'] input").first
    input_box.click()
    input_box.fill("RRF")
    time.sleep(1)
    
    # Check dropdown options in DOM
    options = page.evaluate("""() => {
        return Array.from(document.querySelectorAll('[role="option"], [data-baseweb="menu"] li, [data-baseweb="menu"] div')).map(el => el.innerText.trim()).filter(Boolean);
    }""")
    print("Dropdown options for 'RRF':", options)
    
    # Click the first matching option or press Enter / ArrowDown + Enter
    page.keyboard.press("ArrowDown")
    time.sleep(0.5)
    page.keyboard.press("Enter")
    time.sleep(2)
    
    res = page.locator(".stSuccess").inner_text() if page.locator(".stSuccess").count() > 0 else "None"
    print("Result after ArrowDown + Enter:", res)
    
    browser.close()
