"""Matching one scene across exporter generations.

Two runs picked for comparison are often months apart and name the same recorded
scene differently: release exports use plain names plus a real ``t4dataset_name``,
older pilot exports append a revision hash and store a placeholder dataset id. When
the viewer cannot tell that ``..._PDD001_00945879`` and ``..._PDD001`` are the same
scene, the unmatched run has nothing to narrow by and would fall back to loading its
whole file — every scenario and dataset in it, all stacked onto the same frames,
because ``frame_index`` restarts at 0 per dataset. That is what these tests guard.
"""

import duckdb
import pytest

from lib.scene_name_match import (
    duckdb_like_prefix,
    resolve_scene_name_option,
    scene_name_alias_bases,
    scene_name_predicate,
)

# Names taken from two runs that really are compared side by side on the dashboard:
# the 2026.07 pilot streampetr export and the 2025.11 v4.2.3 release export.
LEGACY_SUITE = "FullPerformance_V1_Fujiyoshida_PDD_5df54148-1687-4255-a52f-84817921c075"
LEGACY_SCENARIO = "FullPerformance_V1_Fujiyoshida_PDD001_00945879"
RELEASE_SUITE = "FullPerformance_V1_Fujiyoshida_PDD"
RELEASE_SCENARIO = "FullPerformance_V1_Fujiyoshida_PDD001"


@pytest.mark.parametrize(
    "name, expected",
    [
        # A uuid suite suffix and a hex scenario suffix both reduce to the release name.
        (LEGACY_SUITE, [RELEASE_SUITE]),
        (LEGACY_SCENARIO, [RELEASE_SCENARIO]),
        # Chained suffixes peel off one at a time, most specific first.
        (
            "Foo_PDD001_00945879_5df54148-1687-4255-a52f-84817921c075",
            ["Foo_PDD001_00945879", "Foo_PDD001"],
        ),
        # Uppercase hex is still a revision hash.
        ("Foo_PDD001_00945ABC", ["Foo_PDD001"]),
    ],
)
def test_revision_suffixes_reduce_to_the_release_name(name, expected):
    assert scene_name_alias_bases(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        # Already-plain names must be left alone, or a scenario would collapse into its
        # suite and match every sibling scenario.
        RELEASE_SUITE,
        RELEASE_SCENARIO,
        "FullPerformance_V1_Omika_PDD002",
        # A meaningful tail is not a hash: too short to be one, or not hex at all.
        "Scene_V1_beef",
        "Scene_V2",
        "Foo_PDD001_shiojiri",
        "Foo_Odaiba",
        # Nothing left to strip down to.
        "00945879",
        "",
    ],
)
def test_meaningful_name_tails_are_never_stripped(name):
    assert scene_name_alias_bases(name) == []


def test_alias_stripping_terminates_on_a_pathological_name():
    """A name that is nothing but chained hashes must not loop or empty out."""
    name = "_".join(["deadbeef"] * 12)
    bases = scene_name_alias_bases(name)
    assert 0 < len(bases) <= 3
    assert all(bases)


@pytest.mark.parametrize(
    "options, linked, expected",
    [
        # Same generation: exact hit.
        ([RELEASE_SCENARIO], RELEASE_SCENARIO, RELEASE_SCENARIO),
        # Link carries a plain name, run offers the hashed one.
        ([LEGACY_SCENARIO], RELEASE_SCENARIO, LEGACY_SCENARIO),
        # Link carries a hashed name, run offers the plain one. This is the direction
        # that used to have no rule at all.
        ([RELEASE_SCENARIO], LEGACY_SCENARIO, RELEASE_SCENARIO),
        ([RELEASE_SUITE], LEGACY_SUITE, RELEASE_SUITE),
        # Ambiguous: several hashed variants and no way to choose, so decline.
        ([LEGACY_SCENARIO, f"{RELEASE_SCENARIO}_7a8b0928"], RELEASE_SCENARIO, ""),
        # Genuinely absent: a different location must not match.
        (["FullPerformance_V1_Omika_PDD001"], LEGACY_SCENARIO, ""),
        # A sibling scenario number must not satisfy the request.
        (["FullPerformance_V1_Fujiyoshida_PDD002"], LEGACY_SCENARIO, ""),
    ],
)
def test_linked_scene_name_resolves_in_both_directions(options, linked, expected):
    assert resolve_scene_name_option(options, linked) == expected


def test_dataset_prefix_disambiguates_between_hashed_variants():
    options = [f"{RELEASE_SCENARIO}_00945879", f"{RELEASE_SCENARIO}_7a8b0928"]
    assert resolve_scene_name_option(options, RELEASE_SCENARIO, "7a8b0928") == options[1]
    # Without the hint there is no basis to choose.
    assert resolve_scene_name_option(options, RELEASE_SCENARIO) == ""


def test_like_prefix_escapes_wildcards_in_scene_names():
    """Scene names are full of underscores, which are LIKE wildcards."""
    assert duckdb_like_prefix("a_b") == "a\\_b%"
    assert duckdb_like_prefix("50%") == "50\\%%"
    assert duckdb_like_prefix("back\\slash") == "back\\\\slash%"


def _scene_rows(con, rows, column, name):
    """Rows selected by the predicate for `name`, against a small in-memory table."""
    con.execute("CREATE OR REPLACE TABLE scenes (scenario_name VARCHAR, suite_name VARCHAR)")
    con.executemany("INSERT INTO scenes VALUES (?, ?)", rows)
    sql, params = scene_name_predicate(column, name)
    return {r[0] for r in con.execute(f"SELECT {column} FROM scenes WHERE {sql}", params).fetchall()}


@pytest.fixture()
def con():
    connection = duckdb.connect()
    yield connection
    connection.close()


def test_predicate_matches_both_generations_of_a_scene(con):
    rows = [
        (RELEASE_SCENARIO, RELEASE_SUITE),
        (LEGACY_SCENARIO, LEGACY_SUITE),
        (f"{RELEASE_SCENARIO}_7a8b0928", LEGACY_SUITE),
    ]
    # Whichever generation is asked for, both are found — that is what lets the run
    # that names the scene differently be narrowed at all.
    for asked in (RELEASE_SCENARIO, LEGACY_SCENARIO):
        assert _scene_rows(con, rows, "scenario_name", asked) == {
            RELEASE_SCENARIO,
            LEGACY_SCENARIO,
            f"{RELEASE_SCENARIO}_7a8b0928",
        }


def test_predicate_does_not_reach_sibling_scenes(con):
    """Widening across naming generations must not widen across scenes."""
    rows = [
        (RELEASE_SCENARIO, RELEASE_SUITE),
        ("FullPerformance_V1_Fujiyoshida_PDD002", RELEASE_SUITE),
        ("FullPerformance_V1_Omika_PDD001", "FullPerformance_V1_Omika_PDD"),
        # A different location whose name merely starts the same way.
        ("FullPerformance_V1_Fujiyoshida_PDD0010", RELEASE_SUITE),
    ]
    assert _scene_rows(con, rows, "scenario_name", LEGACY_SCENARIO) == {RELEASE_SCENARIO}


def test_suite_predicate_keeps_locations_apart(con):
    rows = [
        (RELEASE_SCENARIO, RELEASE_SUITE),
        (RELEASE_SCENARIO, LEGACY_SUITE),
        ("FullPerformance_V1_Omika_PDD001", "FullPerformance_V1_Omika_PDD"),
    ]
    assert _scene_rows(con, rows, "suite_name", LEGACY_SUITE) == {RELEASE_SUITE, LEGACY_SUITE}


def test_predicate_treats_underscores_in_names_literally(con):
    """`_` is a LIKE wildcard, so an unescaped predicate would match neighbours."""
    rows = [
        ("Scene_A_deadbeef", RELEASE_SUITE),
        ("Scene-A-deadbeef", RELEASE_SUITE),
        ("SceneXA_deadbeef", RELEASE_SUITE),
    ]
    assert _scene_rows(con, rows, "scenario_name", "Scene_A_deadbeef") == {"Scene_A_deadbeef"}
