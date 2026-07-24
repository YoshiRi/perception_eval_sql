import json
import pickle
import zipfile
from pathlib import Path

from lib.download_core import extract_archives
from lib.perception_catalog_io import _normalize_loaded_pkl


OLD_TOPIC = "perception.object_recognition.tracking.objects"
NEW_TOPIC = "perception.object_recognition.objects"


class _Frame:
    pass_fail_result = object()


def _make_archive(root: Path, name: str, topic: str) -> Path:
    archive_path = root / f"{name}.zip"
    payload = pickle.dumps([_Frame()])
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(f"{topic}/scene_result.pkl", payload)
        zf.writestr("scenario.yaml", "name: scenario\n")
    return archive_path


def test_extract_archives_auto_detects_new_object_topic_when_old_requested(tmp_path: Path):
    _make_archive(tmp_path, "case_a", NEW_TOPIC)
    case_dir = tmp_path / "case_a"
    case_dir.mkdir()
    (case_dir / "t4_metadata.json").write_text(
        json.dumps({"t4_dataset_id": "dataset-1"}),
        encoding="utf-8",
    )
    warnings = []

    extract_archives(OLD_TOPIC, str(tmp_path), on_warning=warnings.append)

    assert (case_dir / "scene_result.pkl").is_file()
    assert not (case_dir / NEW_TOPIC).exists()
    sidecar = json.loads((case_dir / "t4_metadata.json").read_text(encoding="utf-8"))
    assert sidecar["t4_dataset_id"] == "dataset-1"
    assert sidecar["phase"] == NEW_TOPIC
    assert sidecar["topic_name"] == NEW_TOPIC
    assert any(NEW_TOPIC in warning and OLD_TOPIC in warning for warning in warnings)


def test_normalize_plain_pkl_uses_topic_from_sidecar(tmp_path: Path):
    run_dir = tmp_path / "suite" / "scenario"
    run_dir.mkdir(parents=True)
    pkl_path = run_dir / "scene_result.pkl"
    pkl_path.write_bytes(pickle.dumps([_Frame()]))
    (run_dir / "t4_metadata.json").write_text(
        json.dumps({"topic_name": NEW_TOPIC}),
        encoding="utf-8",
    )

    normalized = _normalize_loaded_pkl([_Frame()], pkl_file=pkl_path)

    assert list(normalized.frame_results.keys()) == [NEW_TOPIC]
