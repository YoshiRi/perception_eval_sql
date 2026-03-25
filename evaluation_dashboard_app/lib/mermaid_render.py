"""Render Mermaid diagrams in Streamlit via Mermaid.js (Streamlit markdown does not run Mermaid)."""

import json
import uuid

import streamlit.components.v1 as components


def render_mermaid(definition: str, *, height: int = 480) -> None:
    """Render a Mermaid diagram inside an HTML iframe (CDN script)."""
    defn_json = json.dumps(definition.strip())
    uid = uuid.uuid4().hex[:12]
    html = f"""
<div id="mermaid-host-{uid}" style="overflow:auto;max-width:100%;padding:0.25rem 0;"></div>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.0/dist/mermaid.min.js"></script>
<script>
(function() {{
  const defn = {defn_json};
  const host = document.getElementById("mermaid-host-{uid}");
  mermaid.initialize({{ startOnLoad: false, theme: "neutral", securityLevel: "loose" }});
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
