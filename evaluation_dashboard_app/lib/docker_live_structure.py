"""
Mermaid source for the Deployment debug Docker tab: same subgraph layout as Readme.md (Help).

Clients → Edge → App Tier → Infrastructure → Workers → Host data, with live container labels.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional


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


def live_containers_mermaid(rows: List[Dict[str, str]]) -> str:
    """
    flowchart LR with subgraphs matching Help / Readme.md:
    Clients, Edge, App Tier, Infrastructure, Workers, Host data — plus live labels per container.
    """
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
    known = set(nginx + st + redis + pg + init + workers)
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

    for i in redis:
        if nl_workers:
            lines.append(f"    {_nid(i)} --> {nl_workers}")

    for i in workers:
        if nl_pg:
            lines.append(f"    {_nid(i)} --> {nl_pg}")
        lines.append(f"    {_nid(i)} --> DR")

    for i in init:
        for j in pg:
            lines.append(f"    {_nid(i)} -.-> {_nid(j)}")

    return "\n".join(lines)
