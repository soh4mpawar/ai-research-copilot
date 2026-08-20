import time
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 720})
    page.goto("http://localhost:8502", wait_until="load")
    time.sleep(3)
    
    print("=== Finding selectbox elements ===")
    inputs = page.evaluate("""() => {
        return Array.from(document.querySelectorAll('input, select, [role="combobox"]')).map(el => ({
            tag: el.tagName,
            type: el.type,
            placeholder: el.placeholder,
            role: el.getAttribute('role'),
            className: el.className
        }));
    }""")
    print("Found inputs:", inputs)
    
    # Locate the select input
    box = page.locator("[data-testid='stSelectbox']").first
    box.click()
    time.sleep(1)
    
    # Type in combobox
    page.keyboard.type("RRF")
    time.sleep(1)
    page.keyboard.press("Enter")
    time.sleep(2)
    
    res1 = page.locator(".stSuccess").inner_text() if page.locator(".stSuccess").count() > 0 else "None"
    print("Result after typing 'RRF' and Enter:", res1)
    
    browser.close()
