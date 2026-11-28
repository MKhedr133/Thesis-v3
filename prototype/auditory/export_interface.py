"""Export the actual local Auditory viewer at 3x resolution for questionnaire review."""
import argparse
from datetime import datetime
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
VIEWER = Path(__file__).with_name("index.html")
LEVEL_CODES = ("0", "2", "6", "10")
SHAP_COMPARISON_EXPORTS = (("all", "all-difficulty-levels"),) + tuple(
    (code, f"level-{level}") for level, code in enumerate(LEVEL_CODES)
)
WIDTH, HEIGHT, SCALE = 1920, 1080, 3

METHODS = {
    "shap": (
        "regression",
        [
            ("01-overview", (".header", "#regression-view .shap-hero")),
            ("02-comparison", ("#shap-compare-details",)),
            ("03-measurements-and-decision", (
                "#regression-view .cd-recorded", "section[aria-labelledby='next-session-heading']"
            )),
        ],
    ),
    "mamdani": (
        "mamdani",
        [
            ("01-overview-and-tree", (".header", "#cd-tree-panel")),
            ("02-rule-activations", ("#mamdani-view .cd-sidebar",)),
            ("03-fuzzy-output", ("#mamdani-view .mamdani-output",)),
            ("04-measurements-and-decision", (
                "#mamdani-view .cd-recorded", "section[aria-labelledby='next-session-heading']"
            )),
        ],
    ),
    "sugeno": (
        "sugeno",
        [
            ("01-overview-and-clusters", (".header", "#sugeno-view .cd-grid")),
            ("02-calculation", ("#sugeno-view .cd-bottom",)),
            ("03-measurements-and-decision", (
                "#sugeno-view .cd-recorded", "section[aria-labelledby='next-session-heading']"
            )),
        ],
    ),
}


def capture(page, destination, selectors):
    """Capture the rendered rectangle, including any intervening layout."""
    clip = page.evaluate(
        """selectors => {
            const rects = selectors.map(selector => {
                const element = document.querySelector(selector);
                if (!element) throw new Error("Missing capture section: " + selector);
                const rect = element.getBoundingClientRect();
                if (!rect.width || !rect.height)
                    throw new Error("Hidden capture section: " + selector);
                return rect;
            });
            const left = Math.floor(Math.min(...rects.map(r => r.left)) + scrollX);
            const top = Math.floor(Math.min(...rects.map(r => r.top)) + scrollY);
            const right = Math.ceil(Math.max(...rects.map(r => r.right)) + scrollX);
            const bottom = Math.ceil(Math.max(...rects.map(r => r.bottom)) + scrollY);
            return {x: left, y: top, width: right - left, height: bottom - top};
        }""",
        list(selectors),
    )
    page.screenshot(path=destination, full_page=True, clip=clip,
                    type="png", scale="device", animations="disabled")


def capture_shap_comparisons(page, folder, name, selectors):
    """Capture each population with the selected session reference unchanged."""
    population = page.locator("#shap-stage-filter")
    original = population.input_value()
    try:
        for value, suffix in SHAP_COMPARISON_EXPORTS:
            population.select_option(value)
            page.locator("#shap-beeswarm svg").wait_for(state="visible")
            page.evaluate("window.scrollTo(0, 0)")
            capture(page, folder / f"{name}-{suffix}.png", selectors)
    finally:
        population.select_option(original)
        page.locator("#shap-beeswarm svg").wait_for(state="visible")
        page.evaluate("window.scrollTo(0, 0)")


def capture_mamdani_rules(page, folder, selectors):
    """Capture every displayed, positively activated rule expanded together."""
    rules = page.locator("#cd-rule-list details.cd-rule")
    original = rules.evaluate_all("nodes => nodes.map(node => node.open)")
    if not original:
        return
    try:
        rules.evaluate_all("nodes => nodes.forEach(node => { node.open = true; })")
        for rule in rules.all():
            rule.locator(".cd-rule-text").wait_for(state="visible")
        page.evaluate("window.scrollTo(0, 0)")
        capture(page, folder / "02-rule-activations-all-expanded.png", selectors)
    finally:
        rules.evaluate_all(
            "(nodes, states) => nodes.forEach((node, i) => { node.open = states[i]; })",
            original,
        )
        page.evaluate("window.scrollTo(0, 0)")


def capture_sugeno_clusters(page, folder, selectors):
    """Capture all three existing cluster comparisons together in an export-only layout."""
    buttons = page.locator("#cd-memberships .cd-membership")
    original = buttons.evaluate_all(
        "nodes => nodes.findIndex(node => node.getAttribute('aria-pressed') === 'true')"
    )
    if buttons.count() != 3 or original < 0:
        raise RuntimeError("Expected three clusters and one selected cluster")
    detail = page.locator("#cd-profile-detail")
    sidebar = page.locator("#sugeno-view .cd-sidebar")
    original_style = sidebar.get_attribute("style")
    try:
        page.evaluate("""() => {
            const host = document.createElement("div");
            host.id = "export-all-clusters";
            host.hidden = true;
            Object.assign(host.style, {
                display: "grid", gridTemplateColumns: "repeat(3,minmax(0,1fr))",
                gap: "14px", alignItems: "start"
            });
            document.getElementById("cd-profile-detail").after(host);
        }""")
        for index in range(3):
            buttons.nth(index).click()
            detail.locator(".cd-feature").first.wait_for(state="visible")
            if detail.get_attribute("aria-label") != f"Cluster {index + 1} recorded measurement comparison":
                raise RuntimeError("Cluster comparison did not update")
            page.evaluate("""index => {
                const host = document.getElementById("export-all-clusters");
                const card = document.createElement("section");
                Object.assign(card.style, {
                    minWidth: "0", border: "1px solid #dce3ea",
                    borderRadius: "10px", padding: "14px", background: "#ffffff"
                });
                const membership = document.querySelectorAll("#cd-memberships .cd-membership")[index]
                    .cloneNode(true);
                membership.removeAttribute("aria-pressed");
                membership.tabIndex = -1;
                const comparison = document.getElementById("cd-profile-detail").cloneNode(true);
                comparison.removeAttribute("id");
                comparison.style.marginTop = "17px";
                comparison.querySelectorAll(".cd-feature > span").forEach(
                    node => { node.style.lineHeight = "1.35"; }
                );
                card.append(membership, comparison);
                host.append(card);
            }""", index)
        page.evaluate("""() => {
            document.querySelector("#sugeno-view .cd-sidebar").style.gridColumn = "1 / -1";
            document.getElementById("cd-memberships").hidden = true;
            document.getElementById("cd-profile-detail").hidden = true;
            document.getElementById("export-all-clusters").hidden = false;
        }""")
        page.evaluate("window.scrollTo(0, 0)")
        capture(page, folder / "01-clusters-all-expanded.png", selectors)
    finally:
        page.evaluate("""style => {
            document.getElementById("export-all-clusters")?.remove();
            document.getElementById("cd-memberships").hidden = false;
            document.getElementById("cd-profile-detail").hidden = false;
            const sidebar = document.querySelector("#sugeno-view .cd-sidebar");
            if (style === null) sidebar.removeAttribute("style");
            else sidebar.setAttribute("style", style);
        }""", original_style)
        buttons.nth(original).click()
        page.evaluate("window.scrollTo(0, 0)")


def prepare_method(page, method):
    tab, _ = METHODS[method]
    page.locator("#tab-" + tab).click()
    panel = page.locator("#" + tab + "-view")
    panel.wait_for(state="visible")
    panel.locator(".cd-recorded").evaluate("el => { el.open = true; }")
    if method == "shap":
        page.locator("#shap-compare-details").evaluate("el => { el.open = true; }")
        page.locator("#shap-beeswarm svg").wait_for(state="visible")
    elif method == "mamdani":
        page.locator("#cd-tree svg").wait_for(state="visible")
        page.locator("#cd-tree-zoom-fit").click()
    else:
        page.locator("#cd-scatter svg").wait_for(state="visible")
    page.evaluate("document.fonts.ready")
    # Do not silently export a horizontally clipped plot or a cropped tree.
    clipped = panel.evaluate(
        """el => [...el.querySelectorAll('.chart,.cd-graphic,.cd-recorded-scroll')]
            .filter(node => node.clientWidth > 0 &&
                (node.scrollWidth > node.clientWidth + 2 ||
                 (node.id === 'cd-tree' && node.scrollHeight > node.clientHeight + 2)))
            .map(node => node.id || node.className)"""
    )
    if clipped:
        raise RuntimeError("Capture would clip: " + ", ".join(clipped))
    page.evaluate("window.scrollTo(0, 0)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path,
                        default=ROOT / "outputs/auditory_abcd_p06_p07_shap_comparison.json")
    parser.add_argument("--participant", required=True, choices=("P06", "P07"))
    parser.add_argument("--level", required=True, type=int, choices=range(4))
    parser.add_argument("--method", choices=("all", *METHODS), default="all")
    parser.add_argument("--browser", choices=("msedge", "chrome", "chromium"),
                        default="msedge")
    args = parser.parse_args()

    source = args.json.resolve(strict=True)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        parser.exit(1, "Playwright is not installed. Run python -m pip show playwright.\n")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = ROOT / "outputs" / "interface_exports" / stamp
    output.mkdir(parents=True, exist_ok=False)
    methods = METHODS if args.method == "all" else (args.method,)

    with sync_playwright() as p:
        launch = {} if args.browser == "chromium" else {"channel": args.browser}
        browser = p.chromium.launch(headless=True, **launch)
        try:
            context = browser.new_context(
                viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=SCALE
            )
            context.route(re.compile(r"^https?://"), lambda route: route.abort())
            page = context.new_page()
            page.goto(VIEWER.as_uri(), wait_until="load")
            page.locator("#data-file").set_input_files(source)
            page.locator("#workspace").wait_for(state="visible")
            page.locator("#participant").select_option(args.participant)
            page.locator("#stage").select_option(LEVEL_CODES[args.level])

            page.locator(".load").evaluate("el => { el.hidden = true; }")

            for method in methods:
                prepare_method(page, method)
                folder = output / method
                folder.mkdir()
                _, sections = METHODS[method]
                for name, selectors in sections:
                    if method == "shap" and name == "02-comparison":
                        capture_shap_comparisons(page, folder, name, selectors)
                    else:
                        capture(page, folder / (name + ".png"), selectors)
                if method == "mamdani":
                    capture_mamdani_rules(page, folder, ("#mamdani-view .cd-sidebar",))
                elif method == "sugeno":
                    capture_sugeno_clusters(page, folder, ("#sugeno-view .cd-sidebar",))
                capture(page, folder / "00-complete-interface.png", (".shell",))
        finally:
            browser.close()

    print("Export complete:", output)


if __name__ == "__main__":
    main()
