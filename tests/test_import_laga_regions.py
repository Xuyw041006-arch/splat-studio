"""Real PNG/JSON/NPY import checks; no model downloads or GPU execution."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import import_laga_regions as importer
from backend.semantic_granularity import build_granularity_targets
from backend.semantic_refinement import SemanticRefinementConfig


def sources(tmp_path, frame="a"):
    seg = np.full((4, 6, 8), -1, np.float32)
    seg[1, 2:4, 3:5] = 0
    seg[2, 1:5, 2:6] = 1
    seg[3, :, :] = 2
    features = np.eye(3, dtype=np.float32)
    np.save(tmp_path/f"{frame}_s.npy", seg)
    np.save(tmp_path/f"{frame}_f.npy", features)
    return {"frame_id": frame, "segmentation_path": f"{frame}_s.npy", "features_path": f"{frame}_f.npy",
            "image_width": 80, "image_height": 60, "coordinate_space": "full_frame"}


def manifest(rows):
    return {"training_frames": [r["frame_id"] for r in rows], "held_out_frames": [],
            "feature_space": "openclip/ViT-B-16/test-checkpoint", "frames": rows}


def test_import_creates_pixels_descriptors_hashes_and_worker_inputs(tmp_path):
    rows = [sources(tmp_path)]
    result = importer.import_regions(manifest(rows), tmp_path/"out", base_dir=tmp_path)
    assert result["training_frame_count"] == 1
    assert result["region_count"] == 3
    output = Path(result["output"])
    records = json.loads((output/"masks.json").read_text())["masks"]
    assert [r["granularity"] for r in records] == ["s", "m", "l"]
    assert [np.asarray(Image.open(output/r["mask_path"])).sum()/255 for r in records] == [4, 16, 48]
    for r in records:
        assert r["mask_sha256"] == importer.sha256(output/r["mask_path"])
        assert len(r["source_features_sha256"]) == 64
        assert r["coordinate_space"] == "full_frame"
        r["mask_path"] = str(output/r["mask_path"])
    built = build_granularity_targets(records, allowed_frames=["a"])
    assert len(built["classes"]) == 3
    config = json.loads((output/"worker-config.json").read_text())
    SemanticRefinementConfig.preset(sampling_schedule=config["sampling_schedule"],
                                    minimum_sweeps=config["minimum_sweeps"]).validate()
    assert config["semantic_granularity"] == "multilevel"


def test_held_out_files_never_opened_and_skipped_audit_explicit(tmp_path):
    train = sources(tmp_path)
    data = manifest([train])
    data["held_out_frames"] = ["test"]
    data["frames"].append({"frame_id": "test", "segmentation_path": "DOES_NOT_EXIST.npy"})
    result = importer.import_regions(data, tmp_path/"out", base_dir=tmp_path)
    assert result["held_out_frame_count_skipped"] == 1
    audit = json.loads((tmp_path/"out/import-audit.json").read_text())
    assert audit["skipped_frames"] == [{"frame_id": "test", "reason": "held_out_not_loaded"}]


def test_entire_split_checked_before_first_np_load(tmp_path, monkeypatch):
    data = manifest([sources(tmp_path)])
    data["frames"].append({"frame_id": "undeclared"})
    def forbidden(*args, **kwargs):
        raise AssertionError("arrays loaded before split validation")
    monkeypatch.setattr(importer.np, "load", forbidden)
    with pytest.raises(ValueError, match="training split"):
        importer.import_regions(data, tmp_path/"out", base_dir=tmp_path)
    assert not (tmp_path/"out").exists()


@pytest.mark.parametrize("mutate,pattern", [
    (lambda d: d.update(held_out_frames=["a"]), "disjoint"),
    (lambda d: d.update(training_frames=["a", "missing"]), "Missing source"),
    (lambda d: d["frames"].append(d["frames"][0]), "Duplicate frame"),
    (lambda d: d["frames"][0].update(coordinate_space="crop"), "full_frame"),
    (lambda d: d["frames"][0].update(crop_box=[0, 0, 8, 6]), "Cropped"),
    (lambda d: d["frames"][0].update(image_width=100), "aspect ratio"),
    (lambda d: d["frames"][0].update(level_mapping={1: "unknown"}), "level_mapping"),
    (lambda d: d["frames"][0].update(region_metadata={"laga-1-99": "ghost"}), "absent regions"),
])
def test_invalid_manifest_never_leaves_partial_deliverable(tmp_path, mutate, pattern):
    data = manifest([sources(tmp_path)])
    mutate(data)
    with pytest.raises(ValueError, match=pattern):
        importer.import_regions(data, tmp_path/"out", base_dir=tmp_path)
    assert not (tmp_path/"out").exists()
    assert list(tmp_path.glob(".out-*")) == []


def test_explicit_parent_metadata_is_carried_without_guessing(tmp_path):
    row = sources(tmp_path)
    row["region_metadata"] = {"laga-1-0": {"label": "handle", "parent_region_id": "laga-2-1",
        "relation": "part_of", "relation_source": "user_region_annotation"}, "laga-2-1": "mug"}
    importer.import_regions(manifest([row]), tmp_path/"out", base_dir=tmp_path)
    records = json.loads((tmp_path/"out/masks.json").read_text())["masks"]
    assert records[0]["parent_region_id"] == "laga-2-1"
    assert records[0]["granularity"] == "s"
    assert records[0]["semantic_name_known"]
    assert not records[2]["semantic_name_known"]


def test_importer_refuses_object_pickle_and_source_oversize(tmp_path, monkeypatch):
    row = sources(tmp_path)
    np.save(tmp_path/"a_f.npy", np.array([{"bad": 1}], object))
    with pytest.raises(ValueError):
        importer.import_regions(manifest([row]), tmp_path/"out", base_dir=tmp_path)
    row = sources(tmp_path)
    monkeypatch.setattr(importer, "MAX_SOURCE_BYTES", 10)
    with pytest.raises(ValueError, match="512 MiB"):
        importer.import_regions(manifest([row]), tmp_path/"out", base_dir=tmp_path)


def test_no_overwrite_existing_result(tmp_path):
    data = manifest([sources(tmp_path)])
    output = tmp_path/"out"
    output.mkdir()
    (output/"keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        importer.import_regions(data, output, base_dir=tmp_path)
    assert (output/"keep.txt").read_text() == "keep"


def test_single_and_manifest_cli(tmp_path, capsys):
    row = sources(tmp_path)
    result = importer.main(["--segmentation", str(tmp_path/"a_s.npy"), "--features", str(tmp_path/"a_f.npy"),
        "--frame", "a", "--feature-space", "clip/test", "--image-width", "80", "--image-height", "60",
        "--output", str(tmp_path/"single")])
    assert result["region_count"] == 3
    path = tmp_path/"sources.json"
    path.write_text(json.dumps(manifest([row])))
    result = importer.main(["--manifest", str(path), "--output", str(tmp_path/"multi")])
    assert result["region_count"] == 3
    assert len(capsys.readouterr().out.strip().splitlines()) == 2
