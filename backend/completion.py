"""Opt-in external learned-completion protocol; no bundled model or shell execution."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np

from .reconstruction import ReconstructionCancelled, ReconstructionError

MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_LOG_BYTES = 10 * 1024 * 1024
MAX_ADDED_GAUSSIANS = 4096


def _argv_from_environment() -> list[str]:
    raw = os.environ.get("SPLAT_COMPLETION_COMMAND", "")
    if not raw:
        raise ReconstructionError("Learned completion is unavailable: no model weights are bundled. Configure SPLAT_COMPLETION_COMMAND as a JSON argv array for an installed local provider.")
    try:
        argv = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReconstructionError("SPLAT_COMPLETION_COMMAND must be a JSON argv array, not a shell command") from exc
    if not isinstance(argv, list) or not argv or len(argv) > 64 or any(not isinstance(arg, str) or not arg or len(arg) > 8192 or "\x00" in arg for arg in argv):
        raise ReconstructionError("SPLAT_COMPLETION_COMMAND must contain 1–64 nonempty string arguments")
    if not any("{request}" in arg for arg in argv) or not any("{output}" in arg for arg in argv):
        raise ReconstructionError("Completion argv must contain {request} and {output} placeholders")
    executable = shutil.which(argv[0])
    if executable is None:
        raise ReconstructionError(f"Completion executable was not found: {Path(argv[0]).name}")
    # Preserve a virtualenv's interpreter symlink; resolving it loses pyvenv.cfg.
    argv[0] = os.path.abspath(executable)
    return argv


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=2)


def validate_completion_output(data: object) -> list[dict]:
    """Validate additional splats; the provider cannot mark predictions observed."""
    if isinstance(data, dict):
        if data.get("output_kind", "additional_gaussians") != "additional_gaussians":
            raise ReconstructionError("Completion output_kind must be additional_gaussians; do not return an entire scene as additive splats")
        gaussians = data.get("gaussians")
    else:
        gaussians = data
    if not isinstance(gaussians, list) or len(gaussians) > MAX_ADDED_GAUSSIANS:
        raise ReconstructionError(f"Completion output must contain at most {MAX_ADDED_GAUSSIANS} additional Gaussians")
    validated = []
    for index, raw in enumerate(gaussians):
        if not isinstance(raw, dict):
            raise ReconstructionError(f"Completion Gaussian {index} must be an object")
        fields = {}
        for key, shape in [("position", (3,)), ("scale", (3,)), ("rotation", (4,)), ("color", (3,)), ("opacity", ())]:
            try:
                value = np.asarray(raw[key], dtype=np.float64)
            except (KeyError, TypeError, ValueError) as exc:
                raise ReconstructionError(f"Completion Gaussian {index} has invalid {key}") from exc
            if value.shape != shape or not np.isfinite(value).all():
                raise ReconstructionError(f"Completion Gaussian {index} has invalid {key}")
            fields[key] = value.tolist()
        if min(fields["scale"]) <= 0 or max(fields["scale"]) > 1e4:
            raise ReconstructionError("Completion scales must lie in (0, 10000]")
        if min(fields["color"]) < 0 or max(fields["color"]) > 1 or not 0 <= fields["opacity"] <= 1:
            raise ReconstructionError("Completion RGB and opacity must lie in [0,1]")
        rotation = np.asarray(fields["rotation"])
        norm = np.linalg.norm(rotation)
        if norm < 1e-8:
            raise ReconstructionError("Completion quaternion must have nonzero norm")
        fields["rotation"] = (rotation / norm).tolist()
        try:
            confidence = float(raw.get("confidence", .25))
        except (TypeError, ValueError) as exc:
            raise ReconstructionError("Completion confidence must be finite") from exc
        if not math.isfinite(confidence):
            raise ReconstructionError("Completion confidence must be finite")
        fields.update({"source": "inferred", "confidence": min(.49, max(0., confidence)), "provenance": "external_completion_provider"})
        # Semantic classes are not accepted as grounded masks through this channel.
        validated.append(fields)
    return validated


def make_provider(image_paths: list[str], output_dir: str | Path, cancelled=None, timeout_seconds: float = 900):
    """Return callable(scene, config) for reconstruction._apply_completion_prior.

    Configuration comes exclusively from the local environment. Image-derived
    text, model output and HTTP request strings never become executable argv.
    """
    command = _argv_from_environment()
    paths = [str(Path(path).resolve()) for path in image_paths]
    if any(not Path(path).is_file() for path in paths):
        raise ReconstructionError("Completion image manifest contains missing local files")
    directory = Path(output_dir).resolve() / "completion"

    def provider(scene: dict, config: dict | None = None) -> list[dict]:
        config = config or {}
        timeout = float(config.get("completion_timeout_seconds", timeout_seconds))
        if not math.isfinite(timeout) or not .05 <= timeout <= 3600:
            raise ReconstructionError("Completion timeout must lie between 0.05 and 3600 seconds")
        if cancelled and cancelled():
            raise ReconstructionCancelled("Completion cancelled before launch")
        directory.mkdir(parents=True, exist_ok=True)
        input_path, request_path, output_path = directory / "input_scene.json", directory / "request.json", directory / "provider_output.json"
        stdout_path, stderr_path = directory / "stdout.log", directory / "stderr.log"
        input_path.write_text(json.dumps(scene, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        manifest = {"protocol": "splat-studio-completion/1", "input_scene": str(input_path), "images": paths, "cameras": scene.get("cameras", []), "coordinate_convention": "OpenCV: x right, y down, z forward; intrinsics in each camera's width/height pixels; world_to_camera is 4x4; scene scale is arbitrary", "gaussian_convention": "position xyz, positive standard-deviation scale xyz, rotation quaternion wxyz, color RGB in [0,1], opacity in [0,1]", "output": str(output_path), "requested_output_kind": "additional_gaussians", "maximum_gaussians": MAX_ADDED_GAUSSIANS, "requirement": "Return only additional predicted splats in the SAME world coordinates and scale. Preserve measured geometry and camera matrices. Unseen surfaces are inferred, never observed."}
        request_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        # Do not accidentally accept a stale result after a failed provider run.
        if output_path.exists():
            output_path.unlink()
        argv = [arg.replace("{request}", str(request_path)).replace("{output}", str(output_path)) for arg in command]
        started = time.monotonic()
        kwargs = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            try:
                process = subprocess.Popen(argv, cwd=directory, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, shell=False, **kwargs)
            except OSError as exc:
                raise ReconstructionError(f"Unable to start configured completion provider: {exc}") from exc
            try:
                while process.poll() is None:
                    if cancelled and cancelled():
                        raise ReconstructionCancelled("Completion cancelled; provider process terminated")
                    if time.monotonic()-started > timeout:
                        raise ReconstructionError(f"Completion provider exceeded its {timeout:g}s timeout")
                    if any(path.exists() and path.stat().st_size > MAX_LOG_BYTES for path in [stdout_path, stderr_path]):
                        raise ReconstructionError("Completion provider exceeded the 10MB log limit")
                    if output_path.exists() and output_path.stat().st_size > MAX_OUTPUT_BYTES:
                        raise ReconstructionError("Completion provider output exceeds the 16MB limit")
                    time.sleep(.05)
            except BaseException:
                _stop(process)
                raise
        if cancelled and cancelled():
            raise ReconstructionCancelled("Completion cancelled")
        if any(path.stat().st_size > MAX_LOG_BYTES for path in [stdout_path, stderr_path]):
            raise ReconstructionError("Completion provider exceeded the 10MB log limit")
        if process.returncode != 0:
            raise ReconstructionError(f"Completion provider exited with code {process.returncode}; inspect {stderr_path}")
        if not output_path.is_file():
            raise ReconstructionError("Completion provider exited successfully but produced no output JSON")
        if output_path.stat().st_size > MAX_OUTPUT_BYTES:
            raise ReconstructionError("Completion provider output exceeds the 16MB limit")
        try:
            data = json.loads(output_path.read_text(encoding="utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ReconstructionError("Completion provider returned invalid JSON") from exc
        result = validate_completion_output(data)
        scene.setdefault("metadata", {})["completion_provider"] = {"protocol": manifest["protocol"], "executable": Path(command[0]).name, "seconds": time.monotonic()-started, "output_count": len(result), "request_path": str(request_path), "stdout_path": str(stdout_path), "stderr_path": str(stderr_path), "bundled_learned_weights": False, "quality_verified": False}
        return result

    return provider
