import time
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    page.goto("http://localhost:8501", wait_until="load")
    time.sleep(5)
    
    print("Triggering dark mode via JS click...")
    page.evaluate("""() => {
        const inputs = Array.from(document.querySelectorAll('input[type="checkbox"]'));
        const darkToggle = inputs.find(i => i.getAttribute('aria-label') === 'Dark Mode');
        if (darkToggle) {
            darkToggle.click();
            console.log('Clicked darkToggle!');
        } else if (inputs.length > 0) {
            inputs[0].click();
        }
    }""")
    time.sleep(4)
    
    page.screenshot(path="scratch/dark_mode_js_test.png", full_page=True)
    print("Saved scratch/dark_mode_js_test.png!")
    browser.close()
