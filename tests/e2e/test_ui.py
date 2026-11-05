import os
import time

from conftest import wait_until

REPORT_TIMEOUT = 60_000


def to_team(page, url, topic="Urban forests", analysts=2):
    page.fill("#topic", topic)
    page.click(f"label[for=c{analysts}]")
    page.click("#start-btn")
    page.wait_for_selector("#view-team:not([hidden])", timeout=20_000)


def open_app(page, url):
    page.goto(url)
    page.wait_for_selector("#examples .chip")


def test_landing_page_is_clean_and_switches_language(page, live_server):
    open_app(page, live_server.url)
    assert page.locator("#examples .chip").count() == 3
    assert "Plant a topic" in page.inner_text("h1")
    assert page.locator("#language option").count() == 2

    page.click("[data-ui-lang=tr]")
    assert "Bir konu ek" in page.inner_text("h1")
    assert page.get_attribute("[data-ui-lang=tr]", "aria-pressed") == "true"
    page.click("[data-ui-lang=en]")
    assert "Plant a topic" in page.inner_text("h1")
    assert page.problems == []


def test_full_research_flow(page, live_server, tmp_path):
    open_app(page, live_server.url)
    page.select_option("#language", "tr")
    to_team(page, live_server.url)

    assert page.locator("#analysts .analyst").count() == 2
    page.fill("#analysts .analyst >> nth=0 >> .analyst__name", "Dr. Edited Name")
    assert page.inner_text("#analysts .analyst >> nth=0 >> .avatar") == "DE"
    page.click("#add-analyst")
    page.click("#approve-btn")
    assert page.locator("#analysts .is-invalid").count() == 2
    page.click("#analysts .analyst >> nth=2 >> .icon-btn")
    assert page.locator("#analysts .analyst").count() == 2

    page.fill("#feedback", "add a skeptic")
    page.click("#regen-btn")
    wait_until(lambda: "revised: add a skeptic" in page.input_value("#analysts .analyst >> nth=0 >> textarea"))

    page.click("#approve-btn")
    page.wait_for_selector(".interview", timeout=10_000)
    assert len(page.locator(".interview__title strong").all_inner_texts()) >= 1
    page.wait_for_selector("#view-report:not([hidden])", timeout=REPORT_TIMEOUT)

    assert page.locator("#report-body h1").inner_text() == "Big Title"
    assert page.locator("#report-body h2").last.inner_text() == "Kaynaklar"
    assert page.locator("a.cite").count() > 0
    ids = page.locator("li.source").evaluate_all("els => els.map(e => e.id)")
    assert ids and ids[0] == "src-1"
    page.click("a.cite >> nth=0")
    wait_until(lambda: page.locator("li.source.is-flash").count() == 1)
    assert "every checked claim" in page.inner_text("#review-summary")
    assert "tokens" in page.inner_text("#report-meta")

    with page.expect_download(timeout=10_000) as info:
        page.click("#download-btn")
    target = tmp_path / "report.md"
    info.value.save_as(target)
    assert target.read_bytes().startswith(b"# Big Title")
    assert info.value.suggested_filename.endswith(".md")

    assert page.locator("#copy-btn, #download-btn, #print-btn, #again-btn").count() == 4
    assert page.locator("#docx-btn, #html-btn, #share-btn, #share-panel").count() == 0

    page.click("#history-btn")
    wait_until(lambda: page.locator("#history-list li").count() >= 1)
    page.click("#history-list li >> nth=0 >> text=Open")
    page.wait_for_selector("#view-report:not([hidden])", timeout=10_000)
    assert page.locator("#report-body h1").inner_text() == "Big Title"
    assert page.problems == []


def test_refreshing_during_a_run_reconnects(page, live_server):
    open_app(page, live_server.url)
    to_team(page, live_server.url)
    page.click("#approve-btn")
    page.wait_for_selector(".interview", timeout=10_000)
    before = page.evaluate("location.hash")

    page.reload()
    page.wait_for_selector("#view-run:not([hidden])", timeout=10_000)
    wait_until(lambda: page.locator(".interview").count() == 2)
    assert page.evaluate("location.hash") == before
    page.wait_for_selector("#view-report:not([hidden])", timeout=REPORT_TIMEOUT)
    assert page.problems == []


def test_stop_then_try_again(page, live_server):
    open_app(page, live_server.url)
    to_team(page, live_server.url, analysts=1)
    live_server.model.delay = 0.5
    try:
        page.click("#approve-btn")
        page.wait_for_selector(".interview", timeout=10_000)
        page.click("#stop-btn")
        page.wait_for_selector("#run-error:not([hidden])", timeout=10_000)
        assert page.inner_text("#h-run") == "Research stopped"
        wait_until(lambda: page.text_content(".interview .badge") == "stopped")
        time.sleep(1.0)
    finally:
        live_server.model.delay = 0.1

    page.click("#retry-btn")
    page.wait_for_selector("#view-report:not([hidden])", timeout=REPORT_TIMEOUT)
    assert page.locator("#report-body h1").inner_text() == "Big Title"


def test_form_has_no_extra_options(page, live_server):
    open_app(page, live_server.url)
    assert page.locator("#options, input[name=depth], input[name=source], #fact-check, #doc-input").count() == 0
    assert page.locator("#topic, #language, input[name=count]").count() == 8


def test_home_button_returns_to_the_landing_page(page, live_server):
    open_app(page, live_server.url)
    assert page.get_attribute("#home-btn", "aria-current") is not None

    page.click("#history-btn")
    page.wait_for_selector("#view-history:not([hidden])")
    assert page.get_attribute("#home-btn", "aria-current") is None
    assert page.get_attribute("#history-btn", "aria-current") is not None
    page.click("#home-btn")
    page.wait_for_selector("#view-topic:not([hidden])")
    assert page.evaluate("location.hash") == ""

    to_team(page, live_server.url, topic="Home button topic", analysts=1)
    live_server.model.delay = 0.3
    try:
        page.click("#approve-btn")
        page.wait_for_selector(".interview", timeout=10_000)
        page.click("#home-btn")
        page.wait_for_selector("#view-topic:not([hidden])")
        assert "keeps running" in page.inner_text("#toast")
        assert page.input_value("#topic") == ""
        assert page.evaluate("location.hash") == ""

        page.click("#history-btn")
        wait_until(lambda: page.locator("#history-list li").count() >= 1)
        page.click("#history-list li >> nth=0 >> text=Open")
        page.wait_for_selector("#view-report:not([hidden])", timeout=REPORT_TIMEOUT)
    finally:
        live_server.model.delay = 0.1
    assert page.locator("#report-body h1").inner_text() == "Big Title"
    assert page.problems == []


def test_mobile_layout_has_no_horizontal_scroll(browser, live_server):
    context = browser.new_context(viewport={"width": 390, "height": 800})
    context.add_init_script("localStorage.setItem('ra-ui-lang', 'en')")
    page = context.new_page()
    open_app(page, live_server.url)
    assert not page.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth")
    context.close()
    assert os.path.exists(os.path.dirname(__file__))
