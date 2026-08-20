import time
import os
from playwright.sync_api import sync_playwright

artifact_dir = r"C:\Users\soham\.gemini\antigravity\brain\5a470592-edee-4883-92fe-3f9d078e5ffb"
typeahead_dropdown_path = os.path.join(artifact_dir, "autocomplete_typeahead_dropdown.png")
custom_query_path = os.path.join(artifact_dir, "autocomplete_custom_query.png")
result_path = os.path.join(artifact_dir, "autocomplete_pipeline_result.png")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 950})
    
    print("1. Loading App...")
    page.goto("http://localhost:8501", wait_until="load")
    page.wait_for_selector("[data-testid='stSelectbox']", timeout=60000)
    time.sleep(3)
    
    print("2. Toggling Dark Mode ON...")
    page.evaluate("""() => {
        const toggle = document.querySelector('[data-testid="stSidebar"] input[type="checkbox"]');
        if (toggle && !toggle.checked) {
            toggle.click();
        }
    }""")
    time.sleep(3)
    
    # 3. Test Type-Ahead Dropdown
    print("3. Testing Type-Ahead Fuzzy Dropdown for 'RRF'...")
    select_box = page.locator("[data-testid='stSelectbox']").first
    select_box.click()
    time.sleep(1)
    
    page.keyboard.type("RRF")
    time.sleep(2)
    
    page.screenshot(path=typeahead_dropdown_path, full_page=False)
    print(f"[OK] Saved Type-Ahead Dropdown Screenshot: {typeahead_dropdown_path}")
    
    # 4. Click the matching option
    print("4. Selecting matching option...")
    matching_opt = page.locator("[role='option']").first
    matching_opt.click()
    time.sleep(2)
    
    # 5. Run the pipeline
    print("5. Clicking Run Research Pipeline...")
    page.locator("button:has-text('Run Research Pipeline')").first.click()
    time.sleep(24)
    
    page.screenshot(path=result_path, full_page=False)
    print(f"[OK] Saved Pipeline Result Screenshot: {result_path}")
    
    # 6. Test Custom Non-Matching Query
    print("6. Testing Custom Non-Matching Query...")
    page.evaluate("() => window.scrollTo(0, 0)")
    time.sleep(1)
    
    # Clear selectbox or click input
    clear_btn = page.locator("[data-testid='stSelectbox'] [role='button'][aria-label='Clear value']")
    if clear_btn.count() > 0:
        clear_btn.click()
        time.sleep(1)
    else:
        select_box.click()
        time.sleep(1)
    
    page.keyboard.type("How does contrastive learning optimize multimodal representations?")
    time.sleep(2)
    
    page.screenshot(path=custom_query_path, full_page=False)
    print(f"[OK] Saved Custom Query Screenshot: {custom_query_path}")
    
    browser.close()
    print("All captures completed successfully!")
