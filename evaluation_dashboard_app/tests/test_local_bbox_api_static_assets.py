"""Static asset serving for the bbox explorer/viewer pages.

A long-running API process must still serve assets added to static/ after it
started: the HTML is re-read from disk per request, so a 404 on a script it
references breaks the whole page rather than just its styling.
"""

import pytest

from backend.local_bbox_api import _asset_content_type, _explorer_html, _viewer_html


def test_known_assets_keep_their_explicit_type():
    assert _asset_content_type("bbox_explorer.css") == "text/css; charset=utf-8"
    assert _asset_content_type("events.js") == "text/javascript; charset=utf-8"


def test_js_and_css_are_served_without_an_allow_list_entry():
    assert _asset_content_type("some_new_module.js") == "text/javascript; charset=utf-8"
    assert _asset_content_type("some_new_sheet.css") == "text/css; charset=utf-8"


def test_other_extensions_and_traversal_are_refused():
    for name in (
        "secrets.env",
        "local_bbox_api.py",
        "config.json",
        "../backend/local_bbox_api.py",
        "..%2fsecrets.js",
        "sub/dir/app.js",
        "",
    ):
        assert _asset_content_type(name) == "", name


@pytest.mark.parametrize("render", [_explorer_html, _viewer_html])
def test_theme_module_is_inlined_into_the_page(render):
    """The palette must ship with the HTML, not as a separate asset request.

    Every canvas renderer calls into it on first paint, so a page served without it
    throws instead of merely losing its colors.
    """
    html = render("")
    assert "/*__BBOX_THEME_JS__*/" not in html, "placeholder was left unsubstituted"
    assert "window.BBoxTheme" in html
    assert 'data-theme' in html
    assert '<script src="bbox_theme.js' not in html, "must not also be linked (double execution)"


@pytest.mark.parametrize("render", [_explorer_html, _viewer_html])
def test_api_base_placeholder_is_substituted(render):
    assert "__API_BASE__" not in render("http://127.0.0.1:8765/")
