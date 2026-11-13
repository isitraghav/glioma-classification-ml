"""Capture demo screenshots of the running Streamlit app (http://localhost:8601).

Usage: python report/tools/capture_demo_screenshots.py report/figs app/artifacts/example_upload_raw.csv
Needs: pip install playwright (uses the installed Microsoft Edge).
"""
import sys, time
from playwright.sync_api import sync_playwright

OUT = sys.argv[1]
EX = sys.argv[2]
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)
    pg = b.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=1.5)
    pg.goto("http://localhost:8601", wait_until="networkidle")
    pg.get_by_text("Class probabilities").wait_for(timeout=120000)
    time.sleep(4)
    pg.screenshot(path=OUT + "/demo_single.png", full_page=True)
    # pick a different patient (a misclassified one is shown later in the 'whole test set' tab)
    pg.get_by_role("tab", name="Whole test set").click()
    pg.get_by_text("Misclassified patients").wait_for(timeout=60000)
    time.sleep(4)
    pg.screenshot(path=OUT + "/demo_testset.png", full_page=True)
    pg.get_by_role("tab", name="Upload expression file").click()
    time.sleep(2)
    pg.locator("input[type=file]").set_input_files(EX)
    pg.get_by_text("Download predictions").wait_for(timeout=60000)
    time.sleep(4)
    pg.screenshot(path=OUT + "/demo_upload.png", full_page=True)
    b.close()
print("ok")
