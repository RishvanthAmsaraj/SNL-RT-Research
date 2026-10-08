"""
ui_check.py -- drive the running app in a real browser, click through every section, and fail on any error.

    streamlit run app.py --server.headless true --server.port 8601      # in one terminal
    python desktop/ui_check.py http://localhost:8601 shots/              # in another (needs: pip install playwright)

Set KINARM_CHROMIUM to a Chromium binary to use it instead of Playwright's own download. Screenshots of every step
are written to the output folder so the layout can be reviewed in light and dark mode.
"""
from __future__ import annotations
import asyncio, os, sys, time
from playwright.async_api import async_playwright

BAD = ("StreamlitAPIException", "Traceback (most recent call last)", "could not be drawn")


async def tour(url: str, out: str, scheme: str, quick: bool) -> list:
    problems = []
    async with async_playwright() as p:
        kw = {"headless": True, "args": ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--hide-scrollbars"]}
        if os.environ.get("KINARM_CHROMIUM"): kw["executable_path"] = os.environ["KINARM_CHROMIUM"]
        browser = await p.chromium.launch(**kw)
        page = await browser.new_page(viewport={"width": 1440, "height": int(os.environ.get("KINARM_UI_HEIGHT", "1000"))}, color_scheme=scheme)
        n = [0]

        async def settle(ms=1500):
            await page.wait_for_timeout(400)
            for _ in range(160):                                   # wait while Streamlit shows its running indicator
                if not await page.locator('[data-testid="stStatusWidget"]').count(): break
                await page.wait_for_timeout(250)
            await page.wait_for_timeout(ms)

        async def shot(name, full=True):
            n[0] += 1; html = await page.content()
            for b in BAD:
                if b in html: problems.append(f"{scheme} {name}: page shows '{b}'")
            await page.screenshot(path=os.path.join(out, f"{scheme}_{n[0]:02d}_{name}.png"), full_page=full)

        async def click(text, nth=0):
            # buttons, and segmented-control options (rendered as buttons in some Streamlit versions, radios/tabs in others)
            for role in ("button", "radio", "tab", "option", "checkbox"):
                loc = page.get_by_role(role, name=text, exact=True)
                if await loc.count() > nth:
                    await loc.nth(nth).click(); await settle(); return True
            problems.append(f"{scheme}: no '{text}' control found"); return False

        await page.goto(url, wait_until="networkidle")
        try: await page.get_by_text("byte-identical", exact=False).first.wait_for(timeout=90000)
        except Exception: problems.append(f"{scheme}: the first page never finished drawing")
        await settle(1500); await shot("e1_overview")
        if quick:
            for step in ("Experiment 2", "Figures", "Open", "Side by side", "Experiment 2", "Models"):
                await click(step); await shot(step.lower().replace(" ", "_"))
            await browser.close(); return problems
        await click("See all figures"); await shot("e1_figures")
        await click("Open"); await shot("e1_figure_detail", full=False)
        await click("Next"); await click("Previous"); await click("Close")
        await click("Compare", nth=1); await shot("side_by_side_pinned")
        await click("LATER"); await shot("side_by_side_later")
        for side in ("L", "R"):                                   # both panes must draw their figure at full pane width
            img = page.locator(f'[class*="st-key-kp-pane-{side}"] img').first
            if await img.count():
                bb = await img.bounding_box()
                if not bb or bb["width"] < 250: problems.append(f"{scheme}: side-by-side pane {side} figure is only {bb and round(bb['width'])} px wide")
            else:
                problems.append(f"{scheme}: side-by-side pane {side} shows no figure")
        await click("Figures"); await shot("compare_figures")
        await click("Tables"); await shot("compare_tables")
        await click("Documents"); await shot("compare_documents")
        await click("Experiment 2"); await shot("e2_overview")
        for view in ("Figures", "Tables", "Documents", "Models", "Run"):
            await click(view); await shot(f"e2_{view.lower()}")
        await click("Experiment 1"); await click("Documents"); await shot("e1_documents")
        await click("Earlier versions and history"); await shot("e1_documents_archive")
        await click("Tables"); await click("Earlier versions"); await shot("e1_tables_archive")
        await click("Run")
        tog = page.get_by_text("Use the bundled sample data", exact=False)
        if await tog.count():
            await tog.first.click(); await settle(2500)
            include = page.get_by_text("Include", exact=True)
            for i in (0, 1):                                       # switch off the two slow analyses for the tour
                if await include.count() > i: await include.nth(i).click(); await settle(800)
            await shot("e1_run_ready")
            await click("Run selected analyses")
            t0 = time.time()
            while time.time() - t0 < 240:
                if await page.get_by_text("Run finished", exact=False).count(): break
                await page.wait_for_timeout(3000)
            await settle(); await shot("e1_run_finished")
            await click("See figures"); await shot("e1_run_figures")
        else:
            problems.append("sample-data toggle not found")
        await browser.close()
    return problems


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
    out = sys.argv[2] if len(sys.argv) > 2 else "ui_shots"; os.makedirs(out, exist_ok=True)
    found = asyncio.run(tour(url, out, "light", quick=False)) + asyncio.run(tour(url, out, "dark", quick=True))
    print("\n".join(found) if found else "UI check passed: every step drew without an error.")
    sys.exit(1 if found else 0)
