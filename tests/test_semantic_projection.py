"""Point-center visibility is matching evidence, not an alpha/accuracy test."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.semantic_projection import projection_support_loader, visible_point_grid


def camera(width=4, height=4):
    return {"width": width, "height": height, "world_to_camera": np.eye(4).tolist(),
            "intrinsics": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}


def test_front_center_occludes_farther_center_across_chunks():
    points = np.array([[2, 2, 2], [1, 1, 1], [3, 3, 3]], float)
    grid = visible_point_grid(points, camera(), 4, 4, chunk_size=1)
    assert grid[1, 1] == 1
    assert np.count_nonzero(grid >= 0) == 1


def test_camera_behind_and_low_opacity_points_excluded():
    points = np.array([[0, 0, -1], [0, 0, 1], [0, 0, 2]], float)
    grid = visible_point_grid(points, camera(), 4, 4, opacity=[1, .01, 1])
    assert grid[0, 0] == 2
    assert not np.isin(grid, [0, 1]).any()


def test_border_continuous_centers_and_last_partial_pixel():
    points = np.array([[-.001, 0, 1], [0, 0, 1], [3.999, 3.999, 1], [4., 2, 1], [1, 4., 1]], float)
    grid = visible_point_grid(points, camera(), 4, 4)
    assert grid[0, 0] == 1
    assert grid[3, 3] == 2
    assert set(grid[grid >= 0]) == {1, 2}


def test_pixel_scale_matches_full_camera_grid():
    cam = camera(width=8, height=8)
    cam["intrinsics"] = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
    points = np.array([[1, 2, 1]], float)
    full, half = visible_point_grid(points, cam, 8, 8), visible_point_grid(points, cam, 4, 4)
    assert full[4, 2] == half[2, 1] == 0


def test_explicit_camera_transform_is_respected():
    cam = camera()
    cam["world_to_camera"][0][3] = -2
    grid = visible_point_grid(np.array([[3, 1, 1.]]), cam, 4, 4)
    assert grid[1, 1] == 0


def test_equal_depth_tie_deterministic_and_chunk_size_independent():
    points = np.array([[1, 1, 1], [1, 1, 1], [2, 2, 2], [3, 3, 1]], float)
    grids = [visible_point_grid(points, camera(), 4, 4, chunk_size=n) for n in (1, 2, 100)]
    assert grids[0][1, 1] == 0
    for grid in grids[1:]:
        np.testing.assert_array_equal(grid, grids[0])


def test_disjoint_region_masks_keep_separate_global_ids():
    points = np.array([[1, 1, 1], [2, 2, 1], [3, 1, 1]], float)
    load = projection_support_loader(points, {"a": camera()})
    left, right = np.zeros((4, 4), bool), np.zeros((4, 4), bool)
    left[1, 1], right[2, 2] = True, True
    assert load({"frame_id": "a"}, left) == [0]
    assert load({"frame_id": "a"}, right) == [1]
    assert load({"frame_id": "a"}, np.ones((4, 4), bool)) == [0, 1, 2]


def test_subsample_keeps_global_ids_and_same_selection_across_views():
    # Model size 250001 triggers stride 2. IDs must not be renumbered after
    # filtering or based on independently sampled per-frame point arrays.
    points = np.zeros((250001, 3), np.float32)
    points[:, 2] = -1
    points[5], points[200000] = [1, 1, 1], [2, 2, 1]
    load = projection_support_loader(points, {"a": camera(), "b": camera()})
    mask = np.ones((4, 4), bool)
    assert load({"frame_id": "a"}, mask) == [200000]
    assert load({"frame_id": "b"}, mask) == [200000]


def test_model_input_rows_never_modified():
    points = np.array([[1, 2, 3], [4, 5, 6]], float)
    original = points.copy()
    visible_point_grid(points, camera(), 4, 4)
    np.testing.assert_array_equal(points, original)


def test_cancel_callback_can_interrupt_support_preparation():
    def stop():
        raise RuntimeError("cancelled")
    loader = projection_support_loader(np.ones((1, 3)), {"a": camera()}, cancelled=stop)
    with pytest.raises(RuntimeError, match="cancelled"):
        loader({"frame_id": "a"}, np.ones((4, 4), bool))


@pytest.mark.parametrize("width,height,chunk", [(0, 4, 1), (4, 0, 1), (4, 4, -1)])
def test_invalid_output_grid_or_chunk_rejected(width, height, chunk):
    with pytest.raises(ValueError):
        visible_point_grid(np.ones((1, 3)), camera(), width, height, chunk_size=chunk)


@pytest.mark.parametrize("points,opacity", [([[1, 2, np.nan]], None), ([[1, 2]], None),
                                          ([[1, 2, 3]], [1, 1])])
def test_invalid_points_and_source_opacity_order_rejected(points, opacity):
    with pytest.raises(ValueError):
        visible_point_grid(np.asarray(points), camera(), 4, 4, opacity=opacity)
