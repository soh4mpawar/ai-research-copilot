import time
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    page.goto("http://localhost:8501", wait_until="load")
    time.sleep(3)
    
    # Enter query in text input
    page.locator("input[aria-label='Enter scientific research question:']").fill("What is RAG?")
    page.locator("button:has-text('Run Research Pipeline')").first.click()
    time.sleep(22)
    
    # Print all button and tab elements
    elements = page.evaluate("""() => {
        return Array.from(document.querySelectorAll('button, [role="tab"], [data-baseweb="tab"]')).map(el => ({
            tag: el.tagName,
            role: el.getAttribute('role'),
            text: el.innerText.trim(),
            className: el.className
        }));
    }""")
    print("Found interactive elements:")
    for el in elements:
        print(f"[{el['tag']}] role={el['role']} text='{el['text']}' class='{el['className']}'")
        
    browser.close()
