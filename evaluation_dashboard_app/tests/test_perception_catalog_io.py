from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lib import perception_catalog_io


class _NewSceneDataFrame:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, object]] = []

    def to_parquet(self, save_dir: Path, compression: object, index: bool = False) -> None:
        self.calls.append((save_dir, compression))
        (save_dir / "current.parquet").write_text("ok", encoding="utf-8")


class _CollectSchemaCompatSceneDataFrame:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, object | None]] = []

    def to_parquet(self, save_dir: Path, compression: object | None = None) -> None:
        self.calls.append((save_dir, compression))
        if compression is not None:
            raise AttributeError("'DataFrame' object has no attribute 'collect_schema'")
        (save_dir / "current.parquet").write_text("ok", encoding="utf-8")


class _OldSceneDataFrame:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def to_parquet(self, save_dir: Path) -> None:
        self.calls.append(save_dir)
        (save_dir / "current.parquet").write_text("ok", encoding="utf-8")


class _NewEmptySceneDataFrame:
    def __init__(self) -> None:
        self.kind = "new"


class _OldEmptySceneDataFrame:
    def __init__(self, current: object | None = None) -> None:
        if current is None:
            raise TypeError("current is required")
        self.current = current


class _NewConcatenateSceneDataFrame:
    def __init__(self) -> None:
        self.used_legacy_kwargs = False

    def concatenate(self, other: object) -> "_NewConcatenateSceneDataFrame":
        return self


class _MixedFutureSceneDataFrame:
    def __init__(self, current: object | None = None, future: object | None = None) -> None:
        self.current = current if current is not None else pd.DataFrame()
        self.future = future

    def concatenate(self, other: object) -> "_MixedFutureSceneDataFrame":
        raise ValueError("Incompatible future flags")


@pytest.mark.parametrize("df_class", [_NewSceneDataFrame, _OldSceneDataFrame])
def test_pkl_archive_to_parquet_supports_old_and_new_to_parquet_signatures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    df_class: type[_NewSceneDataFrame] | type[_OldSceneDataFrame],
) -> None:
    df = df_class()
    monkeypatch.setattr(perception_catalog_io, "_require_analyzer", lambda: None)
    monkeypatch.setattr(
        perception_catalog_io,
        "build_scene_dataframe_from_pkl_dir",
        lambda *args, **kwargs: df,
    )

    parquet_path = perception_catalog_io.pkl_archive_to_parquet(tmp_path)

    assert parquet_path == str(tmp_path / "current.parquet")
    assert (tmp_path / "current.parquet").exists()
    assert df.calls


def test_scene_dataframe_to_parquet_compat_passes_snappy_compression(tmp_path: Path) -> None:
    df = _NewSceneDataFrame()

    perception_catalog_io._scene_dataframe_to_parquet_compat(df, tmp_path)

    assert df.calls[0][1] == perception_catalog_io._default_parquet_compression()


def test_scene_dataframe_to_parquet_compat_retries_collect_schema_error(tmp_path: Path) -> None:
    df = _CollectSchemaCompatSceneDataFrame()

    perception_catalog_io._scene_dataframe_to_parquet_compat(df, tmp_path)

    assert len(df.calls) == 2
    assert df.calls[0][1] == perception_catalog_io._default_parquet_compression()
    assert df.calls[1][1] is None
    assert (tmp_path / "current.parquet").exists()


def test_empty_scene_dataframe_supports_default_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(perception_catalog_io, "SceneDataFrame", _NewEmptySceneDataFrame)

    df = perception_catalog_io._empty_scene_dataframe()

    assert df.kind == "new"


def test_empty_scene_dataframe_falls_back_to_current_dataframe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(perception_catalog_io, "SceneDataFrame", _OldEmptySceneDataFrame)

    df = perception_catalog_io._empty_scene_dataframe()

    assert df.current is not None


def test_concatenate_scene_dataframe_retries_without_ignore_index() -> None:
    df = _NewConcatenateSceneDataFrame()

    assert perception_catalog_io._concatenate_scene_dataframe(df, object(), ignore_index=True) is df


def test_concatenate_scene_dataframe_handles_mixed_future_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(perception_catalog_io, "SceneDataFrame", _MixedFutureSceneDataFrame)
    left = _MixedFutureSceneDataFrame(current=pd.DataFrame({"frame_index": [1]}), future=None)
    right = _MixedFutureSceneDataFrame(
        current=pd.DataFrame({"frame_index": [2]}),
        future=pd.DataFrame({"future_index": [0]}),
    )

    merged = perception_catalog_io._concatenate_scene_dataframe(left, right)

    assert merged.current["frame_index"].tolist() == [1, 2]
    assert merged.future is not None
    assert merged.future["future_index"].tolist() == [0]
