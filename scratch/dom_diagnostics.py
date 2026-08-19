import time
import json
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    page.goto("http://localhost:8501", wait_until="load")
    time.sleep(5)

    print("--- 1. LIGHT MODE DIAGNOSTICS ---")
    light_sidebar_html = page.locator("[data-testid='stSidebar']").inner_html()
    print("Sidebar inner HTML length:", len(light_sidebar_html))

    # Toggle dark mode
    print("--- 2. TOGGLING DARK MODE VIA DOM CLICK ---")
    page.evaluate("""() => {
        const inputs = Array.from(document.querySelectorAll('input[type="checkbox"]'));
        const darkToggle = inputs.find(i => i.getAttribute('aria-label') === 'Dark Mode');
        if (darkToggle) {
            darkToggle.click();
        }
    }""")
    time.sleep(4)

    print("--- 3. INSPECTING LIVE DARK MODE DOM ---")
    diag = page.evaluate("""() => {
        const results = {};
        
        // 1. Inspect App root and containers
        const containers = [
            'body',
            '.stApp',
            '[data-testid="stAppViewContainer"]',
            '[data-testid="stHeader"]',
            'header',
            '.stAppHeader',
            '[data-testid="stToolbar"]',
            'section.main',
            '[data-testid="stSidebar"]',
            '[data-testid="stSidebarContent"]',
            '[data-testid="stSidebarUserContent"]',
            '[data-testid="stBottom"]',
            '[data-testid="stDecoration"]'
        ];
        
        results.containers = {};
        for (const sel of containers) {
            const el = document.querySelector(sel);
            if (el) {
                const style = window.getComputedStyle(el);
                results.containers[sel] = {
                    exists: true,
                    className: el.className,
                    background: style.backgroundColor,
                    backgroundImage: style.backgroundImage,
                    color: style.color,
                    display: style.display,
                    boxShadow: style.boxShadow
                };
            } else {
                results.containers[sel] = { exists: false };
            }
        }
        
        // 2. Inspect Sidebar Radio Items
        const radios = Array.from(document.querySelectorAll('[data-testid="stSidebar"] [data-testid="stRadio"] *'));
        results.sidebar_radio_nodes = radios.map(el => {
            const style = window.getComputedStyle(el);
            return {
                tagName: el.tagName,
                className: el.className,
                innerText: el.innerText,
                textContent: el.textContent,
                color: style.color,
                background: style.backgroundColor,
                fontSize: style.fontSize,
                visibility: style.visibility,
                opacity: style.opacity,
                display: style.display
            };
        });

        // 3. Inspect All Sidebar Labels / Text Nodes
        const sidebarLabels = Array.from(document.querySelectorAll('[data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span'));
        results.sidebar_labels = sidebarLabels.map(el => {
            const style = window.getComputedStyle(el);
            return {
                tagName: el.tagName,
                className: el.className,
                innerText: el.innerText,
                color: style.color,
                opacity: style.opacity
            };
        });

        // 4. Find ANY element on the page that has a light/white background
        const allElements = Array.from(document.querySelectorAll('*'));
        const lightElements = [];
        for (const el of allElements) {
            const style = window.getComputedStyle(el);
            const bg = style.backgroundColor;
            // Check if rgb has values > 200 (light)
            const match = bg.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
            if (match) {
                const r = parseInt(match[1]), g = parseInt(match[2]), b = parseInt(match[3]);
                if (r > 200 && g > 200 && b > 200 && style.display !== 'none' && el.offsetWidth > 0 && el.offsetHeight > 0) {
                    lightElements.push({
                        tagName: el.tagName,
                        className: el.className,
                        id: el.id,
                        testId: el.getAttribute('data-testid'),
                        bg: bg,
                        width: el.offsetWidth,
                        height: el.offsetHeight,
                        top: el.getBoundingClientRect().top,
                        left: el.getBoundingClientRect().left
                    });
                }
            }
        }
        results.light_elements = lightElements.slice(0, 30);

        return results;
    }""")

    with open("scratch/dom_diagnostic_results.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)
        
    print("Saved DOM diagnostics to scratch/dom_diagnostic_results.json!")
    browser.close()
