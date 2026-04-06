"""
Mermaid source for the Deployment debug Docker tab: same subgraph layout as Readme.md (Help).

Clients → Edge → App Tier → T4 dataset server (optional) → Infrastructure → Workers → Host data,
with live container labels. T4 may be a Compose service (e.g. ``t4_server``) or an external HTTP
endpoint from ``T4_VISUALIZER_BASE_URL`` (synthetic node).
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional
from urllib.parse import urlparse


def _by_compose_service(rows: List[Dict[str, str]]) -> Dict[str, List[int]]:
    by: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        svc = (r.get("compose_service") or "").strip()
        if svc and svc != "—":
            by[svc].append(i)
    return by


def _mermaid_plain(s: str, max_len: int) -> str:
    return (s or "")[:max_len].replace('"', "'").replace("\n", " ").replace("#", " ")


def _row_mermaid_label(r: Dict[str, str]) -> str:
    name = _mermaid_plain(r.get("name"), 38)
    stt = _mermaid_plain(r.get("state"), 14)
    svc = _mermaid_plain(r.get("compose_service"), 18) or "—"
    hl = (r.get("health") or "").strip()
    if hl and hl != "—":
        return f"{name}<br/>{stt} · {svc}<br/>{_mermaid_plain(hl, 14)}"
    return f"{name}<br/>{stt} · {svc}"


def _row_class(r: Dict[str, str]) -> str:
    s = (r.get("state") or "").lower()
    if s == "running":
        return "run"
    if s in ("exited", "dead"):
        return "x"
    return "o"


def _nid(i: int) -> str:
    return f"N{i}"


def _nid_list(idxs: List[int]) -> Optional[str]:
    if not idxs:
        return None
    return " & ".join(_nid(i) for i in idxs)


def _is_t4_compose_service(svc: str) -> bool:
    s = (svc or "").strip().lower()
    if not s or s == "—":
        return False
    if s in ("t4_visualizer", "t4_server", "t4_visualizer_server", "t4"):
        return True
    return s.startswith("t4_")


def rowset_has_t4_compose_service(rows: List[Dict[str, str]]) -> bool:
    """True if any listed container is classified as the T4 dataset server (Compose service name)."""
    return any(_is_t4_compose_service(str(r.get("compose_service") or "")) for r in rows)


def _t4_url_display(url: str, *, max_len: int = 52) -> str:
    """Short label for Mermaid (host:port or truncated URL)."""
    u = (url or "").strip()
    if not u:
        return "(not set)"
    try:
        p = urlparse(u)
        if p.netloc:
            out = p.netloc
        else:
            out = u
    except Exception:
        out = u
    out = out.replace('"', "'")
    return out if len(out) <= max_len else out[: max_len - 1] + "…"


T4_SYNTHETIC_NODE = "T4SYN"


def live_containers_mermaid(
    rows: List[Dict[str, str]],
    *,
    t4_visualizer_base_url: Optional[str] = None,
) -> str:
    """
    flowchart LR with subgraphs matching Help / Readme.md:
    Clients, Edge, App Tier, optional T4 dataset server, Infrastructure, Workers, Host data —
    plus live labels per container. External T4 HTTP API appears as a synthetic node when
    ``T4_VISUALIZER_BASE_URL`` is set and no matching Compose service is listed.
    """
    if t4_visualizer_base_url is None:
        t4_visualizer_base_url = os.environ.get("T4_VISUALIZER_BASE_URL", "").strip() or None

    if not rows:
        return 'flowchart LR\n    _empty["No containers in filter"]'

    by = _by_compose_service(rows)
    nginx = sorted(by.get("nginx", []), key=lambda i: rows[i].get("name", ""))
    st: List[int] = []
    for svc in sorted(s for s in by if s.startswith("streamlit")):
        st.extend(sorted(by[svc], key=lambda i: rows[i].get("name", "")))
    redis = sorted(by.get("redis", []), key=lambda i: rows[i].get("name", ""))
    pg = sorted(by.get("postgres", []), key=lambda i: rows[i].get("name", ""))
    init = sorted(by.get("init_db", []), key=lambda i: rows[i].get("name", ""))
    workers = sorted(by.get("worker", []), key=lambda i: rows[i].get("name", ""))
    t4: List[int] = []
    for svc, idxs in by.items():
        if _is_t4_compose_service(svc):
            t4.extend(sorted(idxs, key=lambda i: rows[i].get("name", "")))
    t4 = sorted(set(t4), key=lambda i: rows[i].get("name", ""))
    use_synthetic_t4 = bool(t4_visualizer_base_url) and not t4
    known = set(nginx + st + redis + pg + init + workers + t4)
    other = [i for i in range(len(rows)) if i not in known]

    def node_line(i: int) -> str:
        r = rows[i]
        return f'        {_nid(i)}["{_row_mermaid_label(r)}"]:::{_row_class(r)}'

    lines: List[str] = [
        "flowchart LR",
        "    classDef run fill:#c8e6c9,stroke:#2e7d32",
        "    classDef x fill:#ffcdd2,stroke:#c62828",
        "    classDef o fill:#e0e0e0,stroke:#616161",
        "    classDef syn fill:#e3f2fd,stroke:#1565c0",
        '    subgraph clients ["Clients"]',
        "        BR[Browser]:::syn",
        "    end",
    ]

    if nginx:
        lines.append('    subgraph edge ["Edge"]')
        for i in nginx:
            lines.append(node_line(i))
        lines.append("    end")

    if st:
        lines.append('    subgraph app ["App Tier"]')
        for i in st:
            lines.append(node_line(i))
        lines.append("    end")

    if t4 or use_synthetic_t4:
        lines.append('    subgraph t4tier ["T4 dataset server"]')
        if t4:
            for i in t4:
                lines.append(node_line(i))
        else:
            t4_lab = _mermaid_plain(
                f"T4 visualizer (HTTP)<br/>{_t4_url_display(t4_visualizer_base_url or '')}",
                120,
            )
            lines.append(f'        {T4_SYNTHETIC_NODE}["{t4_lab}"]:::syn')
        lines.append("    end")

    infra = redis + pg + init
    if infra:
        lines.append('    subgraph infra ["Infrastructure"]')
        for i in infra:
            lines.append(node_line(i))
        lines.append("    end")

    if workers:
        lines.append('    subgraph workers ["Workers"]')
        for i in workers:
            lines.append(node_line(i))
        lines.append("    end")

    lines.append('    subgraph volumes ["Host data"]')
    lines.append('        DR[Data root<br/>bind-mounted data]:::syn')
    lines.append("    end")

    if other:
        lines.append('    subgraph misc ["Other"]')
        for i in other:
            lines.append(node_line(i))
        lines.append("    end")

    lines.append("")
    lines.append("    %% Same topology as Readme.md Help")

    nl_nginx = _nid_list(nginx)
    nl_st = _nid_list(st)
    nl_redis = _nid_list(redis)
    nl_pg = _nid_list(pg)
    nl_workers = _nid_list(workers)
    nl_t4: Optional[str]
    if t4:
        nl_t4 = _nid_list(t4)
    elif use_synthetic_t4:
        nl_t4 = T4_SYNTHETIC_NODE
    else:
        nl_t4 = None

    if nl_nginx:
        lines.append(f"    BR --> {nl_nginx}")
        if nl_st:
            for i in nginx:
                lines.append(f"    {_nid(i)} --> {nl_st}")
    elif nl_st:
        lines.append(f"    BR --> {nl_st}")

    for i in st:
        if nl_redis:
            lines.append(f"    {_nid(i)} --> {nl_redis}")
        if nl_pg:
            lines.append(f"    {_nid(i)} --> {nl_pg}")
        if nl_t4:
            lines.append(f"    {_nid(i)} --> {nl_t4}")

    for i in redis:
        if nl_workers:
            lines.append(f"    {_nid(i)} --> {nl_workers}")

    for i in workers:
        if nl_pg:
            lines.append(f"    {_nid(i)} --> {nl_pg}")
        lines.append(f"    {_nid(i)} --> DR")

    if nl_t4:
        if t4:
            for i in t4:
                lines.append(f"    {_nid(i)} --> DR")
        else:
            lines.append(f"    {T4_SYNTHETIC_NODE} --> DR")

    for i in init:
        for j in pg:
            lines.append(f"    {_nid(i)} -.-> {_nid(j)}")

    return "\n".join(lines)
