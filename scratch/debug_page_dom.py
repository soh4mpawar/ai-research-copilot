import time
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 950})
    page.goto("http://localhost:8501", wait_until="load")
    time.sleep(3)
    
    # Print page title and errors
    text = page.locator("body").inner_text()
    print("Page text snippet:\n", text[:600])
    
    elements = page.evaluate("""() => {
        return Array.from(document.querySelectorAll('[data-testid], input, select, button')).map(el => ({
            tag: el.tagName,
            testId: el.getAttribute('data-testid'),
            text: el.innerText ? el.innerText.trim().slice(0, 50) : '',
            className: el.className
        }));
    }""")
    print("Found testids / inputs:")
    for el in elements[:20]:
        print(f"[{el['tag']}] testId={el['testId']} text='{el['text']}'")
        
    browser.close()
