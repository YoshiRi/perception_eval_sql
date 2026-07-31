"""Render Mermaid diagrams in Streamlit via Mermaid.js (Streamlit markdown does not run Mermaid)."""

import json
import uuid

import streamlit.components.v1 as components

from lib.ui.theme import is_dark, tokens


def render_mermaid(definition: str, *, height: int = 480) -> None:
    """Render a Mermaid diagram inside an HTML iframe (CDN script)."""
    defn_json = json.dumps(definition.strip())
    uid = uuid.uuid4().hex[:12]
    # The diagram lives in its own iframe, so the page's CSS custom properties do not
    # reach it: pick Mermaid's own theme and pass raw token values in instead.
    mermaid_theme = "dark" if is_dark() else "neutral"
    host_text = tokens()["text"]
    html = f"""
<div id="mermaid-host-{uid}" style="overflow:auto;max-width:100%;padding:0.25rem 0;color:{host_text};"></div>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.0/dist/mermaid.min.js"></script>
<script>
(function() {{
  const defn = {defn_json};
  const host = document.getElementById("mermaid-host-{uid}");
  mermaid.initialize({{ startOnLoad: false, theme: "{mermaid_theme}", securityLevel: "loose" }});
  const graphId = "mermaid-graph-{uid}";
  mermaid.render(graphId, defn).then(function(res) {{
    host.innerHTML = res.svg;
  }}).catch(function(err) {{
    host.textContent = "Mermaid diagram could not be rendered: " + String(err);
  }});
}})();
</script>
"""
    components.html(html, height=height, scrolling=True)
