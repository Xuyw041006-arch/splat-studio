"""Preflight must retain intermediate views without inventing scene links."""
import cv2
import numpy as np
import pytest

from backend.planning import analyze_images


@pytest.mark.parametrize('groups,expected',[(1,1.0),(2,.5)])
def test_long_capture_chain_and_disconnected_scenes(tmp_path,groups,expected):
    # Global 12-image sampling skips all short-baseline overlaps in this set.
    # Independent strips model two unrelated scenes: they must stay separate.
    per_group=40//groups
    paths=[]
    for group in range(groups):
        rng=np.random.default_rng(120+group)
        strip=rng.integers(0,256,(240,180*(per_group-1)+360),dtype=np.uint8)
        strip=cv2.GaussianBlur(strip,(3,3),.6)
        for index in range(per_group):
            image=strip[:,180*index:180*index+360]
            path=tmp_path/f'{len(paths):04d}.png'
            assert cv2.imwrite(str(path),image)
            paths.append(str(path))
    result=analyze_images(paths)
    assert result['checked_image_count']==40
    assert result['unique_image_count']==40
    assert result['connected_ratio']==expected
    assert len(result['largest_component'])==round(40*expected)
