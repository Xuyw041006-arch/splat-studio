"""Meaningful geometry/rendering integration checks; small enough for CPU CI."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.gaussian_io import SH_C0, write_ply, write_scene
from backend.reconstruction import ReconstructionCancelled, ReconstructionError, _apply_completion_prior, build_sparse_scene, build_imported_scene, reconstruct, render_gaussians, train_gaussians


def test_original_ply_uses_unactivated_parameters(tmp_path):
    scene = {"gaussians": [{"position": [1, 2, 3], "color": [.5, .25, 1], "scale": [.1, .2, .3], "rotation": [1, 0, 0, 0], "opacity": .8}]}
    path = write_ply(scene, tmp_path / "cloud.ply")
    content = Path(path).read_text()
    assert "property float f_dc_0" in content
    values = np.array([float(v) for v in content.split("end_header\n")[1].split()])
    names = [line.split()[-1] for line in content.split("end_header\n")[0].splitlines() if line.startswith("property ")]
    fields = dict(zip(names, values))
    assert fields["f_dc_1"] == pytest.approx(-.25 / SH_C0)
    assert fields["opacity"] == pytest.approx(np.log(4))
    assert [fields[f"scale_{i}"] for i in range(3)] == pytest.approx(np.log([.1, .2, .3]))
    assert [fields[f"f_rest_{i}"] for i in range(45)] == [0] * 45


def test_scene_rejects_nonfinite(tmp_path):
    with pytest.raises(ValueError):
        write_scene({"bad": float("nan")}, tmp_path / "scene.json")


def test_featureless_and_single_image_fail_truthfully(tmp_path):
    from PIL import Image
    paths = []
    for i in range(2):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (80, 80), "white").save(path)
        paths.append(str(path))
    with pytest.raises(ReconstructionError, match="At least two"):
        build_sparse_scene(paths[:1])
    with pytest.raises(ReconstructionError, match="No reliable stereo geometry"):
        build_sparse_scene(paths)


def test_actual_feature_matching_and_triangulation(tmp_path):
    import cv2
    from PIL import Image
    rng = np.random.default_rng(21)
    tiles = []
    for x in np.linspace(-1.25, 1.25, 6):
        for y in np.linspace(-.8, .8, 4):
            texture = rng.integers(0, 255, (50, 50, 3), dtype=np.uint8)
            texture = cv2.GaussianBlur(texture, (3, 3), .5)
            tiles.append((x, y, rng.uniform(3.4, 5.5), texture))
    paths = []
    for image_index, tx in enumerate([0, -.45]):
        image = np.zeros((320, 420, 3), dtype=np.uint8)
        for x, y, z, texture in sorted(tiles, key=lambda tile: -tile[2]):
            left, right = round(378 * (x + tx - .18) / z + 210), round(378 * (x + tx + .18) / z + 210)
            top, bottom = round(378 * (y - .18) / z + 160), round(378 * (y + .18) / z + 160)
            if 0 <= left < right < 420 and 0 <= top < bottom < 320:
                image[top:bottom, left:right] = cv2.resize(texture, (right - left, bottom - top))
        path = tmp_path / f"stereo-{image_index}.png"
        Image.fromarray(image).save(path)
        paths.append(str(path))
    scene = build_sparse_scene(paths, {"focal_ratio": .9})
    assert scene["metadata"]["triangulated_points"] >= 12
    assert scene["metadata"]["registered_views"] == 2
    assert scene["metadata"]["median_reprojection_error_px"] < 3
    assert scene["metadata"]["median_triangulation_angle_deg"] > 1
    assert np.isfinite(scene["points"]).all()
    with pytest.raises(ReconstructionError, match="No reliable stereo geometry"):
        build_sparse_scene([paths[0], paths[0]])


def test_front_to_back_alpha_and_gradients():
    torch = pytest.importorskip("torch")
    points = torch.tensor([[0., 0., 2.], [0., 0., 4.]], requires_grad=True)
    scales = torch.full((2, 3), -1., requires_grad=True)
    rotations = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]], requires_grad=True)
    colors = torch.tensor([[8., -8., -8.], [-8., -8., 8.]], requires_grad=True)
    opacity = torch.full((2,), 5., requires_grad=True)
    k = torch.tensor([[20., 0., 8.], [0., 20., 8.], [0., 0., 1.]])
    image = render_gaussians(points, scales, rotations, colors, opacity, k, torch.eye(4), 16, 16)
    assert image[8, 8, 0] > .98
    assert image[8, 8, 2] < .02
    image.sum().backward()
    for parameter in [points, scales, rotations, colors, opacity]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_anisotropic_covariance_rotates_projected_ellipse():
    torch = pytest.importorskip("torch")
    k = torch.tensor([[25., 0., 8.], [0., 25., 8.], [0., 0., 1.]])
    def render(rotation):
        return render_gaussians(torch.tensor([[0., 0., 2.]]), torch.tensor(np.log([[.3, .04, .04]]), dtype=torch.float32), torch.tensor([rotation], dtype=torch.float32), torch.full((1, 3), 5.), torch.full((1,), 2.), k, torch.eye(4), 16, 16)
    horizontal = render([1, 0, 0, 0])
    vertical = render([2**-.5, 0, 0, 2**-.5])
    assert horizontal[8, 11, 0] > horizontal[11, 8, 0] * 5
    assert vertical[11, 8, 0] > vertical[8, 11, 0] * 5
    assert torch.allclose(horizontal.permute(1, 0, 2), vertical, atol=1e-5)


def test_known_camera_training_reduces_photometric_loss():
    torch = pytest.importorskip("torch")
    positions = np.array([[-.2, -.1, 2.], [.25, .12, 2.3]], np.float32)
    truth_colors = np.array([[.85, .15, .2], [.15, .65, .8]], np.float32)
    k = np.array([[30., 0., 16.], [0., 30., 16.], [0., 0., 1.]], np.float32)
    cameras, images = [], []
    for i, translation in enumerate([0., .15]):
        transform = np.eye(4, dtype=np.float32)
        transform[0, 3] = translation
        with torch.no_grad():
            image = render_gaussians(torch.tensor(positions), torch.tensor(np.log(np.full((2, 3), .13, np.float32))), torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]]), torch.tensor(np.log(truth_colors / (1 - truth_colors))), torch.tensor([2., 2.]), torch.tensor(k), torch.tensor(transform), 32, 32, torch.zeros(3)).numpy()
        images.append(image)
        cameras.append({"image_index": i, "width": 32, "height": 32, "intrinsics": k.tolist(), "world_to_camera": transform.tolist()})
    result = train_gaussians(positions, np.full((2, 3), .45, np.float32), cameras, images, {"device": "cpu", "image_size": 32, "iterations": 55, "initial_scales": [[.13]*3]*2, "background": [0, 0, 0]})
    assert result["metrics"]["final_l1"] < result["metrics"]["initial_l1"] * .6
    assert all(g["source"] == "observed" for g in result["gaussians"])


def test_cancellation_and_completion_provenance():
    with pytest.raises(ReconstructionCancelled):
        train_gaussians(np.array([[0, 0, 2]], np.float32), np.ones((1, 3), np.float32), [{"unused": True}], [], {"device": "cpu"}, cancelled=lambda: True)
    scene = {"gaussians": [{"position": [0, 0, 2], "scale": [.1]*3, "rotation": [1, 0, 0, 0], "color": [.5]*3, "opacity": .8, "source": "observed"}], "metadata": {"warnings": []}}
    _apply_completion_prior(scene, {"completion": "bounded_prior"})
    assert len(scene["gaussians"]) == 2
    assert scene["gaussians"][0]["source"] == "observed"
    assert scene["gaussians"][1]["source"] == "inferred"
    assert scene["gaussians"][1]["confidence"] < .5
    with pytest.raises(ReconstructionError, match="no completion_provider"):
        _apply_completion_prior(scene, {"completion": "learned"})


def test_priority_mask_allocates_more_gaussians_to_selected_region():
    pytest.importorskip("torch")
    x = np.r_[np.linspace(-.8, -.1, 50), np.linspace(.1, .8, 50)]
    points = np.column_stack([x, np.zeros(100), np.full(100, 2)]).astype(np.float32)
    camera = {"image_index": 0, "image_path": "fixture.png", "width": 32, "height": 32, "intrinsics": [[25, 0, 16], [0, 25, 16], [0, 0, 1]], "world_to_camera": np.eye(4).tolist()}
    mask = np.zeros((32, 32), np.float32); mask[:, :16] = 1
    result = train_gaussians(points, np.full((100, 3), .5, np.float32), [camera], [np.zeros((32, 32, 3), np.float32)], {"device": "cpu", "iterations": 1, "max_gaussians": 10, "image_size": 32, "priority_masks": {"fixture.png": mask}, "priority_strength": 10})
    assert sum(index < 50 for index in result["selected_indices"]) >= 8
    assert result["metrics"]["priority_weighted"] is True


def test_imported_camera_bootstrap_refines_without_fake_triangulation(tmp_path):
    from PIL import Image
    path = tmp_path / "external.png"; Image.new("RGB", (32, 32)).save(path)
    imported = {"gaussians": [{"position": [0, 0, 2], "color": [.6, .2, .2], "scale": [.08, .04, .04], "rotation": [1, 0, 0, 0], "opacity": .6, "source": "inferred", "confidence": .9}], "cameras": [{"image_name": path.name, "width": 64, "height": 64, "intrinsics": [[50, 0, 32], [0, 50, 32], [0, 0, 1]], "world_to_camera": np.eye(4).tolist()}]}
    scene_path = tmp_path / "external_scene.json"; scene_path.write_text(json.dumps(imported))
    result = reconstruct([str(path)], str(tmp_path / "result"), {"device": "cpu", "iterations": 2, "image_size": 32, "imported_scene_path": str(scene_path)})
    assert result["metadata"]["triangulated_points"] == 0
    assert result["metadata"]["imported_points"] == 1
    assert result["metadata"]["strategy"] == "imported_geometry_refinement"
    assert result["scene"]["gaussians"][0]["source"] == "inferred"
    assert result["scene"]["gaussians"][0]["confidence"] <= .49 + 1e-6
    assert result["scene"]["cameras"][0]["intrinsics"][0][0] == 25
    imported["cameras"][0]["world_to_camera"][0][0] = 2
    with pytest.raises(ReconstructionError, match="rigid"):
        build_imported_scene([str(path)], imported, {})
