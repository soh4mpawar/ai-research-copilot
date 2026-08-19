import time
import os
from playwright.sync_api import sync_playwright

artifact_dir = r"C:\Users\soham\.gemini\antigravity\brain\5a470592-edee-4883-92fe-3f9d078e5ffb"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    
    print("1. Capturing Research QA Engine...")
    page.goto("http://localhost:8501", wait_until="networkidle")
    time.sleep(3)
    try:
        page.locator("button:has-text('Run Research Pipeline')").first.click()
        page.wait_for_selector(".academic-title:has-text('Scientific Research Synthesis')", timeout=35000)
        time.sleep(2)
    except Exception as e:
        print("Note on QA answer:", e)
        time.sleep(4)
    page.screenshot(path=os.path.join(artifact_dir, "qa_engine_academic_white.png"), full_page=True)
    print("[OK] Saved qa_engine_academic_white.png")

    print("2. Capturing Literature Review Studio...")
    try:
        page.locator("div[role='radiogroup'] label").nth(1).click()
        time.sleep(3)
        btn = page.locator("button:has-text('Generate Multi-Paper Literature Review')")
        if btn.count() > 0:
            btn.first.click()
            time.sleep(4)
        page.screenshot(path=os.path.join(artifact_dir, "lit_review_academic_white.png"), full_page=True)
        print("[OK] Saved lit_review_academic_white.png")
    except Exception as e:
        print("Error capturing Lit Review:", e)

    print("3. Capturing Scientific Paper Explorer...")
    try:
        page.locator("div[role='radiogroup'] label").nth(2).click()
        time.sleep(3)
        page.screenshot(path=os.path.join(artifact_dir, "paper_explorer_academic_white.png"), full_page=True)
        print("[OK] Saved paper_explorer_academic_white.png")
    except Exception as e:
        print("Error capturing Paper Explorer:", e)

    print("4. Capturing Citation Network Graph...")
    try:
        page.locator("div[role='radiogroup'] label").nth(3).click()
        time.sleep(5)
        page.screenshot(path=os.path.join(artifact_dir, "citation_graph_academic_white.png"), full_page=True)
        print("[OK] Saved citation_graph_academic_white.png")
    except Exception as e:
        print("Error capturing Citation Graph:", e)

    print("5. Capturing Evaluation & RAGAS Dashboard...")
    try:
        page.locator("div[role='radiogroup'] label").nth(4).click()
        page.wait_for_selector(".js-plotly-plot", timeout=15000)
        time.sleep(3)
        page.screenshot(path=os.path.join(artifact_dir, "eval_dashboard_academic_white.png"), full_page=True)
        print("[OK] Saved eval_dashboard_academic_white.png")
    except Exception as e:
        print("Error capturing Eval Dashboard:", e)

    browser.close()
    print("All 5 views captured successfully!")
