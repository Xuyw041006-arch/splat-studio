import json
from pathlib import Path
import zipfile
import numpy as np
import pytest
from backend.semantic_hierarchy import prepare_hierarchy_records, hierarchy_vocabulary
from backend.semantic_granularity import build_granularity_targets, cross_view_confidence
from backend.semantic_profiles import resolve_semantic_profile
from backend.semantic_package import write_semantic_package, import_semantic_package, sha256
from backend.gaussian_io import read_ply
from tests.test_semantic_worker import make_ply


def test_all_quality_profiles_execute_hierarchy_and_confidence_with_distinct_budgets():
    profiles=[resolve_semantic_profile({'mode':m,'semantic_budget':'mode'}) for m in ('fast','balanced','fine')]
    assert [p['semantic_steps'] for p in profiles]==[400,1200,2400]
    assert [p['semantic_view_mode'] for p in profiles]==['sampled','sampled','all']
    assert [p['semantic_max_views'] for p in profiles[:2]]==[12,24]
    assert all(p['automatic_hierarchy'] and p['cross_view_confidence'] for p in profiles)


def hierarchy_rows():
    rows=[]
    for frame in ('a','b'):
        for name,level,sl in [('stuffed bear','object',slice(1,9)),('bear nose','part',slice(3,5))]:
            mask=np.zeros((10,10),bool);mask[sl,sl]=True
            rows.append(dict(frame_id=frame,mask_id=name,label=name,level=level,mask=mask,confidence=.8))
    return rows


def test_named_part_and_observed_scene_collection_require_repeated_geometry():
    rows=prepare_hierarchy_records(hierarchy_rows())
    built=build_granularity_targets(rows,{'geometry_binding':{'source_ply_sha256':'a'*64,'gaussian_count':100},'min_shared_points':2},support_loader=lambda row,mask:np.flatnonzero(mask).tolist())
    assert len(built['classes'])==3
    assert {edge['relation'] for edge in built['hierarchy']['edges']}=={'part_of','member_of'}
    assert all(edge['support_view_count']==2 for edge in built['hierarchy']['edges'])
    audit=built['diagnostics']['cross_view_confidence']
    assert audit['executed'] and all(r['comparison_views']==1 and r['agreement']==1 for r in audit['observations'])
    # No duplicate or invented part when the teacher doesn't observe one.
    rows=prepare_hierarchy_records([r for r in hierarchy_rows() if r['label']=='stuffed bear'])
    assert not any(r['label']=='bear nose' for r in rows)
    assert 'bear nose' in hierarchy_vocabulary(['stuffed bear'])


def test_view_disagreement_reduces_confidence_but_occlusion_is_unknown():
    a=dict(key=('a','r'),frame='a',support=frozenset(range(10)),confidence=.8)
    b=dict(key=('b','r'),frame='b',support=frozenset(range(6)),confidence=.9)
    result=cross_view_confidence([a,b],{'a':frozenset(range(10)),'b':frozenset(range(10))})
    assert result[a['key']]['agreement']==pytest.approx(.6)
    assert result[a['key']]['effective_confidence']<.8
    occluded=cross_view_confidence([a,b],{'a':frozenset(range(10)),'b':frozenset(range(6))})
    assert occluded[a['key']]['effective_confidence']==pytest.approx(.8)
    single=cross_view_confidence([a],{'a':a['support']})[a['key']]
    assert single['agreement'] is None and single['effective_confidence']==pytest.approx(.28)


def package(tmp_path):
    ply=tmp_path/'scene.ply';identity=make_ply(ply,6)
    folder=tmp_path/'sem';folder.mkdir()
    labels=['whole','part'];probs=np.array([[.9,.8]]*3+[[.9,.1]]*3,dtype=np.float32)
    common=dict(classes=np.asarray(labels),source_ply_sha256=np.asarray(identity))
    artifacts={}
    for kind in ('probabilities','closed_probabilities','membership','closed_membership'):
        file=folder/(kind+'.npz')
        if 'probabilities' in kind:np.savez_compressed(file,probabilities=probs,**common)
        else:np.savez_compressed(file,**{f'class_{i}':probs[:,i]>=.5 for i in range(2)},**common)
        artifacts[kind]={'path':file.name,'sha256':sha256(file)}
    catalog=dict(status='completed',source_ply_sha256=identity,gaussian_count=6,classes=labels,artifacts=artifacts,
        semantic_granularity='multilevel',hierarchy_executed=True,cross_view_confidence={'executed':True},
        hierarchy_edges=[dict(parent='whole',child='part',relation='part_of',source='test_observations',observation_criteria_met=True)],
        class_metadata=[dict(label='whole',granularity='object',cross_view_confidence=.8),dict(label='part',granularity='part',cross_view_confidence=.7)])
    (folder/'catalog.json').write_text(json.dumps(catalog))
    archive=write_semantic_package(folder,tmp_path/'semantic.zip')
    return ply,archive


def test_paired_import_restores_hierarchy_and_keeps_every_source_gaussian(tmp_path):
    ply,archive=package(tmp_path);original=read_ply(ply)
    restored=import_semantic_package(original,archive,tmp_path/'import')
    assert len(restored['gaussians'])==6
    by_label={o['label']:o for o in restored['objects']}
    assert by_label['part']['parent_id']==by_label['whole']['id']
    assert by_label['part']['cross_view_confidence']==.7
    assert restored['metadata']['semantic_import_validated']
    assert all(a['sh']==b['sh'] for a,b in zip(original['gaussians'],restored['gaussians']))
    original['metadata']['source_ply_sha256']='b'*64
    with pytest.raises(ValueError,match='不匹配'):import_semantic_package(original,archive,tmp_path/'wrong')
    assert not (tmp_path/'wrong').exists()


def test_zip_paths_cannot_escape_destination(tmp_path):
    archive=tmp_path/'malformed.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('../outside.json','{}')
    with pytest.raises(ValueError,match='越界'):import_semantic_package({},archive,tmp_path/'import')
    assert not (tmp_path/'outside.json').exists()


def test_single_channel_confidence_changes_actual_loss_and_gradient():
    import torch
    from backend.semantic_refinement import SemanticRefinementConfig, _segmentation_loss
    cfg=SemanticRefinementConfig.preset()
    target=torch.tensor([[[1.,0.],[1.,0.]]]);weights=torch.ones_like(target)
    gradients=[];losses=[]
    for confidence in (1.,.2):
        prediction=torch.full_like(target,.6,requires_grad=True)
        loss,_,_=_segmentation_loss(prediction,target,weights,torch.tensor([confidence]),cfg)
        loss.backward();losses.append(loss.item());gradients.append(prediction.grad)
    assert losses[1]==pytest.approx(losses[0]*.2)
    assert torch.allclose(gradients[1],gradients[0]*.2)
