"""Exercise the real subprocess protocol with explicitly synthetic test providers."""
import json
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.completion import make_provider, validate_completion_output
from backend.reconstruction import ReconstructionCancelled, ReconstructionError


def _fixture(tmp_path, monkeypatch, body):
    script = tmp_path / "test_provider.py"
    script.write_text(body)
    image = tmp_path / "image.png"; image.write_bytes(b"fixture placeholder; subprocess does not decode image")
    monkeypatch.setenv("SPLAT_COMPLETION_COMMAND", json.dumps([sys.executable, str(script), "{request}", "{output}"]))
    return [str(image)]


def test_missing_model_and_shell_command_are_explicit(monkeypatch, tmp_path):
    monkeypatch.delenv("SPLAT_COMPLETION_COMMAND", raising=False)
    with pytest.raises(ReconstructionError, match="no model weights are bundled"):
        make_provider([], tmp_path)
    monkeypatch.setenv("SPLAT_COMPLETION_COMMAND", "python model.py && echo unsafe")
    with pytest.raises(ReconstructionError, match="JSON argv"):
        make_provider([], tmp_path)


def test_actual_provider_process_receives_scene_and_cameras(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch, '''import json,sys
from pathlib import Path
request=json.loads(Path(sys.argv[1]).read_text())
scene=json.loads(Path(request['input_scene']).read_text())
assert request['protocol']=='splat-studio-completion/1'
assert len(scene['cameras'])==1 and scene['cameras']==request['cameras']
assert Path(request['images'][0]).exists()
Path(sys.argv[2]).write_text(json.dumps({'output_kind':'additional_gaussians','gaussians':[{'position':[0,0,2],'scale':[.1,.2,.1],'rotation':[2,0,0,0],'color':[.4,.5,.6],'opacity':.8,'source':'observed','confidence':1.0,'object_id':'unsupported-semantic-claim'}]}))
''')
    scene = {"gaussians": [], "cameras": [{"intrinsics": [[10,0,5],[0,10,5],[0,0,1]], "world_to_camera": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}], "metadata": {}}
    result = make_provider(paths, tmp_path)(scene, {})
    assert result[0]["source"] == "inferred"
    assert result[0]["confidence"] == .49
    assert result[0]["rotation"] == [1, 0, 0, 0]
    assert "object_id" not in result[0]
    assert scene["metadata"]["completion_provider"]["bundled_learned_weights"] is False


def test_timeout_and_cancel_terminate_provider(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch, "import time\ntime.sleep(20)\n")
    started = time.monotonic()
    with pytest.raises(ReconstructionError, match="timeout"):
        make_provider(paths, tmp_path, timeout_seconds=.1)({"gaussians": [], "cameras": []}, {})
    assert time.monotonic()-started < 3
    started = time.monotonic()
    with pytest.raises(ReconstructionCancelled):
        make_provider(paths, tmp_path, cancelled=lambda: time.monotonic()-started > .1)({"gaussians": [], "cameras": []}, {})
    assert time.monotonic()-started < 3


def test_output_validation_and_failed_process(tmp_path, monkeypatch):
    with pytest.raises(ReconstructionError, match="additional_gaussians"):
        validate_completion_output({"output_kind": "replace_scene", "gaussians": []})
    with pytest.raises(ReconstructionError, match="invalid position"):
        validate_completion_output([{"position": [float("nan"), 0, 0]}])
    paths = _fixture(tmp_path, monkeypatch, "raise SystemExit(7)\n")
    with pytest.raises(ReconstructionError, match="code 7"):
        make_provider(paths, tmp_path)({"gaussians": [], "cameras": []}, {})
    # A zero exit code is not sufficient; an output artifact is required.
    paths = _fixture(tmp_path, monkeypatch, "pass\n")
    with pytest.raises(ReconstructionError, match="no output JSON"):
        make_provider(paths, tmp_path)({"gaussians": [], "cameras": []}, {})
