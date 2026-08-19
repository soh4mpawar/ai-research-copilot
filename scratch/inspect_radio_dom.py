import time
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    page.goto("http://localhost:8501", wait_until="load")
    time.sleep(5)

    radio_html = page.evaluate("""() => {
        const rg = document.querySelector('[data-testid="stSidebar"] [role="radiogroup"]');
        return rg ? rg.outerHTML : "None";
    }""")
    print("Radio HTML:")
    print(radio_html)
    browser.close()
