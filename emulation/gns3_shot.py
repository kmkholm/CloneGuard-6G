"""Screenshot of the live GNS3 project (web UI) for Figure 3(b): figures/gns3_shot.png.
Same approach as DAPPER (Playwright Chromium with the server credentials).  python gns3_shot.py [project_id]"""
import json, os, sys
from playwright.sync_api import sync_playwright
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.abspath(os.path.join(HERE, "..", "figures"))
INI = os.path.join(os.environ["APPDATA"], "GNS3", "2.2", "gns3_server.ini")
PW = [l.split("=", 1)[1].strip() for l in open(INI) if l.startswith("password")][0]
PID = sys.argv[1] if len(sys.argv) > 1 else json.load(open(os.path.join(HERE, "project_ids.json")))["project_id"]
BASE = "http://localhost:3080/static/web-ui"; VW = 2300


def add_server(page):
    page.goto(BASE + "/servers", wait_until="networkidle")
    if page.locator("text=Add server").count() == 0:
        return
    page.click("text=Add server"); page.wait_for_selector("mat-dialog-container")
    inputs = page.locator("mat-dialog-container input"); inputs.nth(0).fill("local"); inputs.nth(1).fill("localhost")
    page.locator("mat-dialog-container mat-select").last.click(); page.locator("mat-option", has_text="Basic").click()
    inputs = page.locator("mat-dialog-container input"); inputs.nth(inputs.count() - 2).fill("admin"); inputs.nth(inputs.count() - 1).fill(PW)
    page.locator("mat-dialog-container button", has_text="Add").click(); page.wait_for_timeout(1000)


with sync_playwright() as p:
    br = p.chromium.launch(headless=True)
    ctx = br.new_context(viewport={"width": VW, "height": 1100}, device_scale_factor=2, http_credentials={"username": "admin", "password": PW})
    page = ctx.new_page(); add_server(page)
    page.goto(f"{BASE}/server/1/project/{PID}", wait_until="networkidle"); page.wait_for_selector("svg", timeout=60000); page.wait_for_timeout(6000)
    try: page.locator("text=Close").first.click(timeout=1500)
    except Exception: pass
    page.mouse.click(VW - 41, 30); page.wait_for_timeout(1500)
    page.evaluate("document.body.style.zoom = 0.8"); page.wait_for_timeout(2500)
    page.screenshot(path=os.path.join(OUT, "gns3_shot.png"), full_page=False); br.close()
print("saved", os.path.join(OUT, "gns3_shot.png"))
