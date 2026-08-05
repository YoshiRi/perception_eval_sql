"""Matching suite/scenario names across exporter generations.

The same recorded scene is named differently depending on which evaluator export
produced the parquet:

* release exports use plain names — ``FullPerformance_V1_Fujiyoshida_PDD`` /
  ``FullPerformance_V1_Fujiyoshida_PDD001`` — and carry the real dataset in
  ``t4dataset_name``;
* older pilot exports append a revision hash to both — ``..._PDD_5df54148-1687-4255-a52f-84817921c075``
  / ``..._PDD001_00945879`` — and store a single placeholder ``t4dataset_name`` for
  every dataset.

Comparing two such runs side by side means recognising that those names denote the
same scene, in whichever direction the comparison runs. Getting this wrong is not a
cosmetic issue: a viewer that cannot pin a run to one scene ends up loading every
scenario in the file, and since ``frame_index`` restarts at 0 for each dataset, all
of them stack onto the same frames.

The rule is deliberately narrow. Only a trailing segment that *looks* like a
revision id — a uuid, or a hex string of six or more characters — is strippable, so a
meaningful tail such as ``_PDD001`` is never removed. Candidates are generated, not
asserted: a name no run contains simply never matches, which makes generating one
alias too many harmless and missing one the real failure.
"""

from __future__ import annotations

import re
from typing import Any, List, Sequence

__all__ = [
    "MAX_SCENE_NAME_STRIPS",
    "SCENE_NAME_SUFFIX_RE",
    "clean_scene_text",
    "duckdb_like_prefix",
    "resolve_scene_name_option",
    "scene_name_alias_bases",
    "scene_name_predicate",
]

SCENE_NAME_SUFFIX_RE = re.compile(
    r"_(?:"
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    r"|[0-9a-fA-F]{6,}"
    r")$"
)

# Some exports append both a scenario hash and a suite uuid; a couple of rounds covers
# every layering seen so far without letting a pathological name loop.
MAX_SCENE_NAME_STRIPS = 3


def clean_scene_text(value: Any) -> str:
    """Normalise a name that may have arrived as a null-ish placeholder."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "nan", "<na>"} else text


def duckdb_like_prefix(text: str) -> str:
    """Escape a data string for DuckDB ``LIKE`` and append a trailing wildcard."""
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
        + "%"
    )


def scene_name_alias_bases(name: Any) -> List[str]:
    """Progressively shorter forms of a suite/scenario name, most specific first.

    Returns only the stripped forms, never the name itself, and an empty list for a
    name that carries no revision suffix.
    """
    bases: List[str] = []
    current = clean_scene_text(name)
    for _ in range(MAX_SCENE_NAME_STRIPS):
        stripped = SCENE_NAME_SUFFIX_RE.sub("", current)
        if not stripped or stripped == current:
            break
        bases.append(stripped)
        current = stripped
    return bases


def scene_name_predicate(column: str, name: Any) -> tuple[str, List[Any]]:
    """A SQL predicate matching one scene name in both suffix directions.

    Covers the name as given, names extending it with a suffix (plain name -> hashed
    name), and each hash-stripped base plus its extensions (hashed name -> plain
    name). Every alternative stays inside the same scene family, so this widens which
    naming generations match without widening the scene.

    Returns ``(sql, params)`` where the SQL is already parenthesised and safe to join
    with ``AND``.
    """
    text = clean_scene_text(name)
    clauses = [f"{column} = ?", f"{column} LIKE ? ESCAPE '\\'"]
    params: List[Any] = [text, duckdb_like_prefix(f"{text}_")]
    for base in scene_name_alias_bases(text):
        clauses.append(f"{column} = ?")
        params.append(base)
        clauses.append(f"{column} LIKE ? ESCAPE '\\'")
        params.append(duckdb_like_prefix(f"{base}_"))
    return "(" + " OR ".join(clauses) + ")", params


def resolve_scene_name_option(
    options: Sequence[str],
    base_name: Any,
    suffix_prefix: str = "",
) -> str:
    """Pick the name from ``options`` that denotes the same scene as ``base_name``.

    Tries, in order: the name as given; the name extended by ``suffix_prefix`` when
    the caller knows the dataset id prefix; a sole option extending the name; then the
    same two steps for each hash-stripped base. Returns ``""`` when nothing matches
    unambiguously, so the caller can leave the current selection alone rather than
    guess.
    """
    base = clean_scene_text(base_name)
    if not base:
        return ""
    option_list = list(options)
    if base in option_list:
        return base
    if suffix_prefix:
        wanted = f"{base}_{suffix_prefix}"
        if wanted in option_list:
            return wanted
    matches = [opt for opt in option_list if str(opt).startswith(f"{base}_")]
    if len(matches) == 1:
        return matches[0]
    for alias in scene_name_alias_bases(base):
        if alias in option_list:
            return alias
        alias_matches = [opt for opt in option_list if str(opt).startswith(f"{alias}_")]
        if len(alias_matches) == 1:
            return alias_matches[0]
    return ""
