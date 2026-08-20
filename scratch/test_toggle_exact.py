import time
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    page.goto("http://localhost:8501", wait_until="load")
    time.sleep(3)
    
    print("Initial body bg:", page.evaluate("() => window.getComputedStyle(document.body).backgroundColor"))
    
    # Click via dispatchEvent or element click
    page.evaluate("""() => {
        const toggle = document.querySelector('[data-testid="stSidebar"] input[type="checkbox"]');
        if (toggle) {
            toggle.click();
        }
    }""")
    time.sleep(3)
    
    print("After toggle body bg:", page.evaluate("() => window.getComputedStyle(document.body).backgroundColor"))
    browser.close()
