from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.gaussian_io import read_ply, write_ply


def test_imported_ply_has_unknown_provenance_and_no_fabricated_confidence(tmp_path):
    original = {"gaussians": [{"position": [1, 2, 3], "color": [.4, .5, .6],
        "scale": [.1, .2, .3], "rotation": [1, 0, 0, 0], "opacity": .8,
        "source": "observed", "confidence": .97}]}
    path = write_ply(original, tmp_path / "cloud.ply")
    imported = read_ply(path)
    assert imported["gaussians"][0]["source"] == "unknown"
    assert "confidence" not in imported["gaussians"][0]
    assert "semantic_confidence" not in imported["gaussians"][0]
    assert imported["metadata"]["preview_sh_degree"] == 3
    assert imported["metadata"]["sh_layout"] == 'channel_major_including_dc'
    assert imported["metadata"]["warnings"]
