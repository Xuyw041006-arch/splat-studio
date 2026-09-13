"""Saved RGB artifacts can be serialized without repeating training."""
import json
import pytest
from backend import upstream


def test_finalize_preserves_saved_ply_sh_and_priority_receipt(tmp_path, monkeypatch):
    run=tmp_path/'run';ply=run/'model/point_cloud/iteration_22000/point_cloud.ply'
    ply.parent.mkdir(parents=True);ply.write_bytes(b'completed-model-identity')
    scene={'gaussians':[{'sh':list(range(48)),'sh_degree':3}], 'cameras':[], 'metadata':{}}
    monkeypatch.setattr(upstream,'read_ply',lambda path,max_points:scene if path==ply and max_points is None else pytest.fail('Unexpected model or sampling'))
    monkeypatch.setattr(upstream,'run_logged',lambda *a,**k:pytest.fail('RGB training was invoked'))
    image=tmp_path/'image.jpg';image.write_bytes(b'training-photo')
    evidence={'status':'completed','added_gaussians':17}
    (run/'priority-detail').mkdir();(run/'priority-detail/evidence.json').write_text(json.dumps(evidence))
    cameras=[{'image_name':'image.jpg','mask_pixel_coordinates_verified':True}]
    result=upstream.finalize_upstream_model([image],run,{'mode':'fine','priority_masks':True,'priority_refinement':'detail','priority_objects':['mug']},root=tmp_path,source_model=tmp_path/'source',verified_cameras=cameras,priority_camera_names=['image.jpg'])
    saved=json.loads((run/'scene.json').read_text())
    assert ply.read_bytes()==b'completed-model-identity'
    assert saved['gaussians'][0]['sh']==list(range(48))
    assert saved['metadata']['priority_enhancement']['added_gaussians']==17
    assert saved['cameras']==cameras
    assert result['ply_path']==str(ply)


def test_finalize_does_not_claim_missing_priority_evidence(tmp_path,monkeypatch):
    image=tmp_path/'image.jpg';image.write_bytes(b'photo')
    monkeypatch.setattr(upstream,'read_ply',lambda *a,**k:{'metadata':{},'gaussians':[]})
    with pytest.raises(RuntimeError,match='验收证据'):
        upstream.finalize_upstream_model([image],tmp_path/'run',{'mode':'fine','priority_masks':True,'priority_refinement':'detail','priority_objects':['mug']},root=tmp_path,source_model=tmp_path)
    assert not (tmp_path/'run/scene.json').exists()
