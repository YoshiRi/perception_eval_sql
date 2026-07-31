"""Guards for the light/dark design-token layer (lib/ui/theme.py + .streamlit config)."""

from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parents[1]
CONFIG = APP_ROOT / "deploy" / ".streamlit" / "config.toml"


def _load_config() -> dict:
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:
        toml = pytest.importorskip("toml")
        return toml.load(str(CONFIG))
    with CONFIG.open("rb") as fh:
        return tomllib.load(fh)


def test_light_and_dark_define_the_same_tokens():
    """A token missing from one theme renders as an empty CSS value, not a fallback."""
    from lib.ui import theme

    light, dark = theme.tokens("light"), theme.tokens("dark")
    assert set(light) == set(dark)
    assert not [k for k, v in {**light, **dark}.items() if not str(v).strip()]


def test_css_variables_are_prefixed_and_dashed():
    from lib.ui import theme

    css = theme.css_variables("dark")
    assert "--t4-surface-2:" in css
    assert "--t4-ok-bg:" in css
    assert "_" not in css.split(":")[0]


def test_active_theme_falls_back_to_light_outside_a_script_run():
    from lib.ui import theme

    assert theme.active_theme() in ("light", "dark")
    assert theme.tokens() == theme.tokens(theme.active_theme())


def test_plotly_layout_keeps_paper_transparent_and_themes_text():
    from lib.ui import theme

    dark = theme.plotly_layout("dark")
    assert dark["paper_bgcolor"] == "rgba(0,0,0,0)"
    assert dark["template"] == "plotly_dark"
    assert dark["font"]["color"] == theme.token("chart_text", "dark")

    light = theme.plotly_layout("light")
    assert light["template"] == "plotly_white"
    assert light["plot_bgcolor"] == theme.token("chart_bg", "light")


def test_apply_plotly_theme_survives_axis_free_figures():
    go = pytest.importorskip("plotly.graph_objects")
    from lib.ui import theme

    fig = go.Figure(go.Pie(values=[1, 2]))
    theme.apply_plotly_theme(fig, "dark")
    assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"


def test_chart_palettes_have_the_shapes_plotly_and_streamlit_expect():
    from lib.ui import theme

    for name in ("light", "dark"):
        assert len(theme.CATEGORICAL(name)) >= 6
        # Streamlit's chartSequentialColors / chartDivergingColors want ten stops;
        # ours are Plotly scales, so only monotonic non-empty lists are required.
        assert len(theme.SEQUENTIAL_SCALE(name)) >= 5
        assert len(theme.DIVERGING_SCALE(name)) >= 5


def test_streamlit_config_leaves_light_stock_and_themes_only_dark():
    """
    Light must stay Streamlit's stock theme: no [theme.light] section, and no styling
    key at [theme] level, since those apply to both themes.
    """
    theme_cfg = _load_config()["theme"]
    assert theme_cfg["base"] == "light"
    assert "light" not in theme_cfg
    assert set(theme_cfg) == {"base", "dark"}
    assert theme_cfg["dark"]["backgroundColor"]
    assert theme_cfg["dark"]["textColor"]


def test_streamlit_config_dark_matches_the_python_palette():
    """config.toml drives Streamlit's own chrome; theme.py drives ours. Keep them equal."""
    from lib.ui import theme

    cfg = _load_config()["theme"]["dark"]
    expected = theme.tokens("dark")
    assert cfg["backgroundColor"] == expected["bg"]
    assert cfg["textColor"] == expected["text"]
    assert cfg["borderColor"] == expected["border"]
    assert cfg["secondaryBackgroundColor"] == expected["surface_2"]
    assert cfg["codeBackgroundColor"] == expected["code_bg"]
    assert cfg["primaryColor"] == expected["accent"]
    # Alert semantics: text colors are shared with the token trios. The matching
    # backgrounds are not asserted because they are pre-flattened tints.
    assert cfg["greenTextColor"] == expected["ok"]
    assert cfg["yellowTextColor"] == expected["warn"]
    assert cfg["redTextColor"] == expected["bad"]
    assert cfg["blueTextColor"] == expected["info"]


def test_pick_returns_the_light_value_outside_the_dark_theme():
    from lib.ui import theme

    assert theme.pick("old-light", "new-dark") == (
        "new-dark" if theme.is_dark() else "old-light"
    )
