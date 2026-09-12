"""Original 3DGS PLY channel layout and preview round-trip, CPU only."""
from pathlib import Path
import sys

import numpy as np
from plyfile import PlyData, PlyElement
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.gaussian_io import SH_C0, read_ply, write_ply


def splat(degree):
    n = (degree+1)**2
    return {"position": [1., 2., 3.], "scale": [.1, .2, .3], "rotation": [1, 0, 0, 0],
            "color": [.3, .4, .5], "opacity": .7, "sh_degree": degree,
            "sh": np.concatenate([np.arange(n)*.125 + channel for channel in range(3)]).tolist()}


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_channel_major_sh_matches_original_ply_properties_and_round_trip(tmp_path, degree):
    original = splat(degree)
    path = tmp_path/"source.ply"
    write_ply({"gaussians": [original]}, path, sh_degree=degree)
    vertex = PlyData.read(path)["vertex"]
    n = (degree+1)**2
    for channel in range(3):
        assert vertex[f"f_dc_{channel}"][0] == pytest.approx(original["sh"][channel*n])
        for j in range(n-1):
            assert vertex[f"f_rest_{channel*(n-1)+j}"][0] == pytest.approx(original["sh"][channel*n+j+1])
    loaded = read_ply(path)
    assert loaded["metadata"]["preview_sh_degree"] == degree
    assert loaded["metadata"]["sh_layout"] == "channel_major_including_dc"
    np.testing.assert_allclose(loaded["gaussians"][0]["sh"], original["sh"], atol=1e-6)
    write_ply(loaded, tmp_path/"again.ply", sh_degree=degree)
    np.testing.assert_allclose(read_ply(tmp_path/"again.ply")["gaussians"][0]["sh"], original["sh"], atol=1e-6)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_clamped_preview_rgb_does_not_destroy_negative_or_high_raw_dc(tmp_path, degree):
    original = splat(degree)
    n = (degree+1)**2
    original["sh"][0], original["sh"][n], original["sh"][2*n] = -10, 10, 0
    write_ply({"gaussians": [original]}, tmp_path/"source.ply", sh_degree=degree)
    scene = read_ply(tmp_path/"source.ply")
    assert scene["gaussians"][0]["color"] == [0., 1., .5]
    write_ply(scene, tmp_path/"again.ply", sh_degree=degree)
    np.testing.assert_allclose(read_ply(tmp_path/"again.ply")["gaussians"][0]["sh"], original["sh"])


def test_dc_only_round_trip_preserves_raw_source_coefficient(tmp_path):
    original = splat(0)
    original["sh"] = [-10, 10, 0]
    write_ply({"gaussians": [original]}, tmp_path/"source.ply", sh_degree=0)
    loaded = read_ply(tmp_path/"source.ply")
    assert loaded["gaussians"][0]["color"] == [0., 1., .5]
    write_ply(loaded, tmp_path/"again.ply", sh_degree=0)
    vertex = PlyData.read(tmp_path/"again.ply")["vertex"]
    np.testing.assert_allclose([vertex[f"f_dc_{i}"][0] for i in range(3)], original["sh"])


def test_explicit_dc_fallback_uses_clamped_colors_and_no_higher_terms(tmp_path):
    original = splat(3)
    original["sh"][0], original["sh"][16] = -10, 10
    write_ply({"gaussians": [original]}, tmp_path/"source.ply", sh_degree=3)
    loaded = read_ply(tmp_path/"source.ply", include_sh=False)
    assert loaded["metadata"]["source_sh_degree"] == 3
    assert loaded["metadata"]["preview_sh_degree"] == 0
    assert loaded["gaussians"][0]["color"][:2] == [0., 1.]
    assert "sh" not in loaded["gaussians"][0]
    write_ply(loaded, tmp_path/"fallback.ply", sh_degree=3)
    vertex = PlyData.read(tmp_path/"fallback.ply")["vertex"]
    assert all(vertex[f"f_rest_{j}"][0] == 0 for j in range(45))


def test_degree_upgrade_zero_pads_each_color_independently(tmp_path):
    original = splat(1)
    write_ply({"gaussians": [original]}, tmp_path/"source.ply", sh_degree=3)
    actual = np.asarray(read_ply(tmp_path/"source.ply")["gaussians"][0]["sh"]).reshape(3, 16)
    np.testing.assert_allclose(actual[:, :4], np.asarray(original["sh"]).reshape(3, 4))
    assert not actual[:, 4:].any()


def test_sh_sampling_retains_original_global_row_indices(tmp_path):
    write_ply({"gaussians": [splat(2) for _ in range(8)]}, tmp_path/"source.ply", sh_degree=2)
    scene = read_ply(tmp_path/"source.ply", max_points=3)
    assert [g["source_index"] for g in scene["gaussians"]] == [0, 3, 7]
    assert scene["metadata"]["preview_sampled"]


@pytest.mark.parametrize('budget', [None, 'full'])
def test_full_import_preserves_every_row_and_coefficient(tmp_path, budget):
    originals=[splat(3) for _ in range(8)]
    for i, gaussian in enumerate(originals):gaussian['sh'][47]=i/16
    path=tmp_path/'all.ply';write_ply({'gaussians':originals},path)
    loaded=read_ply(path,max_points=budget)
    assert [g['source_index'] for g in loaded['gaussians']]==list(range(8))
    assert not loaded['metadata']['preview_sampled']
    np.testing.assert_allclose([g['sh'] for g in loaded['gaussians']],[g['sh'] for g in originals])


def test_full_resource_limit_checks_header_before_decoding_vertices(tmp_path):
    path=tmp_path/'oversize.ply'
    path.write_text('ply\nformat ascii 1.0\nelement vertex 2000001\nproperty float x\nend_header\n')
    with pytest.raises(ValueError,match='不会自动抽样'):
        read_ply(path)


@pytest.mark.parametrize('budget',[True,False,0,-1,2_000_001,1.5,'auto'])
def test_invalid_preview_budget_rejected_before_reading(tmp_path,budget):
    with pytest.raises(ValueError,match='max_points'):
        read_ply(tmp_path/'not-read.ply',max_points=budget)


def test_scene_serialization_failure_preserves_previous_file(tmp_path):
    from backend.gaussian_io import write_scene
    path=tmp_path/'scene.json';path.write_text('{"prior":true}')
    with pytest.raises(ValueError):write_scene({'invalid':float('nan')},path)
    assert path.read_text()=='{"prior":true}'
    assert list(tmp_path.iterdir())==[path]


@pytest.mark.parametrize("alter", [lambda g: g.update(sh_degree=4), lambda g: g.update(sh_degree=True),
                                   lambda g: g.update(sh=g["sh"][:-1]),
                                   lambda g: g["sh"].__setitem__(5, float("nan"))])
def test_malformed_export_coefficients_rejected(tmp_path, alter):
    original = splat(2)
    alter(original)
    with pytest.raises(ValueError):
        write_ply({"gaussians": [original]}, tmp_path/"bad.ply", sh_degree=2)


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_nonfinite_import_dc_rejected_for_every_degree(tmp_path, degree):
    path = tmp_path/"bad.ply"
    write_ply({"gaussians": [splat(degree)]}, path, sh_degree=degree)
    vertex = PlyData.read(path)["vertex"].data.copy()
    vertex["f_dc_0"][0] = np.nan
    PlyData([PlyElement.describe(vertex, "vertex")]).write(path)
    with pytest.raises(ValueError, match="finite"):
        read_ply(path)


def test_incomplete_original_rest_layout_is_rejected(tmp_path):
    source = tmp_path/"bad.ply"
    write_ply({"gaussians": [splat(1)]}, source, sh_degree=1)
    vertex = PlyData.read(source)["vertex"].data
    dtype = [("f_rest_99" if name == "f_rest_8" else name, vertex.dtype[name]) for name in vertex.dtype.names]
    data = np.empty(len(vertex), dtype=dtype)
    for old, (new, _) in zip(vertex.dtype.names, dtype):
        data[new] = vertex[old]
    PlyData([PlyElement.describe(data, "vertex")]).write(source)
    with pytest.raises(ValueError, match="incomplete"):
        read_ply(source)
