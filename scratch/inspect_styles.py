import time
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    page.goto("http://localhost:8501", wait_until="load")
    time.sleep(5)

    print("Clicking dark toggle...")
    page.evaluate("""() => {
        const inputs = Array.from(document.querySelectorAll('input[type="checkbox"]'));
        const darkToggle = inputs.find(i => i.getAttribute('aria-label') === 'Dark Mode');
        if (darkToggle) darkToggle.click();
    }""")
    time.sleep(4)

    style_info = page.evaluate("""() => {
        const styles = Array.from(document.querySelectorAll('style'));
        return styles.map(s => s.innerHTML.substring(0, 300));
    }""")
    
    print("Found styles count:", len(style_info))
    for idx, s in enumerate(style_info):
        print(f"\n--- Style {idx} ---")
        print(s)

    browser.close()
