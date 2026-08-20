import time
import json
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    
    print("1. Loading App...")
    page.goto("http://localhost:8501", wait_until="load")
    time.sleep(3)
    
    print("2. Toggling Dark Mode...")
    page.evaluate("""() => {
        const toggle = document.querySelector('[data-testid="stSidebar"] input[type="checkbox"]');
        if (toggle && !toggle.checked) {
            toggle.click();
        }
    }""")
    time.sleep(3)
    
    print("3. Running QA query...")
    page.locator("button:has-text('Run Research Pipeline')").first.click()
    time.sleep(22)
    
    print("4. Switching to Retrieval Pipeline Transparency tab...")
    page.locator("[data-testid='stTabs'] [data-baseweb='tab']").nth(1).click()
    time.sleep(4)
    
    print("5. Inspecting DataFrame DOM structure...")
    df_info = page.evaluate("""() => {
        const dfs = Array.from(document.querySelectorAll('[data-testid="stDataFrame"], [data-testid="stDataFrameResizable"], [data-testid="stTable"], .dvn-scroller, canvas, [data-testid="stDataFrameContainer"]'));
        return dfs.map(el => {
            const cs = window.getComputedStyle(el);
            return {
                tagName: el.tagName,
                className: el.className,
                testId: el.getAttribute('data-testid'),
                bg: cs.backgroundColor,
                color: cs.color,
                border: cs.border,
                outerHTML: el.outerHTML.slice(0, 300)
            };
        });
    }""")
    
    print(f"Found {len(df_info)} dataframe/table DOM elements:")
    for d in df_info:
        print(f"[{d['tagName']}] testId={d['testId']} class='{d['className']}' -> BG: {d['bg']}, Color: {d['color']}")
        
    browser.close()
