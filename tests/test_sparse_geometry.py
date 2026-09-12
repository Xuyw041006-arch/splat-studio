"""Geometric behavior checks independent of pretrained model quality."""
import copy
from types import SimpleNamespace
import sys
import cv2
import numpy as np
import pytest
from backend import reconstruction as reconstruction
from backend.sparse_geometry import (image_coverage, project_points, extend_registered_geometry,
                                     bundle_adjust, align_visible_completion)


def camera(index, tx=0., path=None):
    transform=np.eye(4);transform[0,3]=tx
    return {'image_index':index,'image_path':path or f'{index}.png','width':320,'height':240,
            'intrinsics':[[200.,0,160.],[0,200.,120.],[0,0,1]],'world_to_camera':transform.tolist()}


def truth_scene():
    rng=np.random.default_rng(19)
    points=np.column_stack([rng.uniform(-1,1,40),rng.uniform(-.8,.8,40),rng.uniform(4,6,40)]).astype(np.float32)
    cameras=[camera(0),camera(1,-.6)]
    pixels=[project_points(points,c)[0] for c in cameras]
    return {'points':points,'colors':np.full_like(points,.5),'confidence':np.full(len(points),.8,np.float32),
            'sources':['observed']*len(points),'observations':[{str(j):pixels[j][i].tolist() for j in range(2)} for i in range(len(points))],
            'cameras':cameras,'images':[np.zeros((240,320,3),np.uint8) for _ in cameras],
            'metadata':{'warnings':[],'triangulated_points':len(points)}}


def learned_scene(sfm):
    rng=np.random.default_rng(31)
    dense=np.column_stack([rng.uniform(-.4,.4,30),rng.uniform(-.3,.3,30),rng.uniform(4.3,5.4,30)]).astype(np.float32)
    rotation=cv2.Rodrigues(np.array([.05,-.02,.2]))[0];scale=1.7;translation=np.array([.3,-.2,.1])
    inverse=lambda p:(p-translation)@rotation/scale
    # Extra point is far outside the verified image/support region.
    dense=np.vstack([dense,[100.,100.,4.]])
    pixels=[project_points(dense,c)[0] for c in sfm['cameras']]
    samples=[{'track_index':i,'frame_index':frame,'point':inverse(p).tolist(),
              'pixel':sfm['observations'][i][str(frame)],'confidence':.4}
             for i,p in enumerate(sfm['points']) for frame in range(2)]
    return {'points':inverse(dense).astype(np.float32),'colors':np.full_like(dense,.6),'confidence':np.full(len(dense),.4),
            'sources':['inferred']*len(dense),'cameras':copy.deepcopy(sfm['cameras']),
            'observations':[{str(j):pixels[j][i].tolist() for j in range(2)} for i in range(len(dense))],
            'images':sfm['images'],'metadata':{'warnings':[]},'alignment_samples':samples}


def test_border_support_is_reported_without_eroding_valid_pixels():
    result=image_coverage([[0,0],[319,239],[160,120],[-.2,5],[320.1,10]],(240,320))
    assert result['count']==3
    assert result['border_fraction']==pytest.approx(2/3)
    assert result['grid_fraction']>0


def test_pnp_new_view_adds_geometry_not_present_in_seed_pair():
    scene=truth_scene();points=scene['points'];seed_count=24
    cameras=[camera(0),camera(1,-.6),camera(2,-1.2)]
    images=scene['images']+[np.zeros((240,320,3),np.uint8)]
    visible=[list(range(seed_count)),list(range(len(points))),list(range(len(points)))]
    features=[]
    for cam,ids in zip(cameras,visible):
        uv,_=project_points(points[ids],cam)
        features.append(([cv2.KeyPoint(float(p[0]),float(p[1]),1) for p in uv],None))
    def matches(i,j):
        return [cv2.DMatch(visible[i].index(pi),visible[j].index(pi),0.) for pi in set(visible[i])&set(visible[j])]
    output=extend_registered_geometry(points[:seed_count],scene['colors'][:seed_count],scene['confidence'][:seed_count],
        scene['observations'][:seed_count],cameras[:2],{0:{i:i for i in range(seed_count)},1:{i:i for i in range(seed_count)}},
        images,[f'{i}.png' for i in range(3)],[np.asarray(c['intrinsics']) for c in cameras],features,matches,
        reconstruction._triangulate,reconstruction._camera,config={})
    assert len(output['cameras'])==3
    assert output['diagnostics']['added_points']==len(points)-seed_count
    assert np.allclose(output['points'][seed_count:],points[seed_count:],atol=1e-4)
    assert all(set(track)=={'1','2'} for track in output['observations'][seed_count:])
    assert all(len(track)>=2 for track in output['observations'])
    assert len(output['colors'])==len(output['points'])==len(output['confidence'])


def test_bundle_adjustment_reduces_real_reprojection_error_and_preserves_seed_gauge():
    scene=truth_scene();truth=scene['points'];cameras=scene['cameras']+[camera(2,-1.2)]
    observations=[dict(t,**{'2':project_points(truth,cameras[2])[0][i].tolist()}) for i,t in enumerate(scene['observations'])]
    noisy=truth+np.random.default_rng(42).normal(0,.012,truth.shape)
    estimated=copy.deepcopy(cameras);estimated[2]['world_to_camera'][0][3]+=.03
    refined,after,diagnostic=bundle_adjust(noisy,estimated,observations,{0,1},{'ba_max_nfev':35})
    assert diagnostic['accepted']
    assert diagnostic['reprojection_median_after_px']<diagnostic['reprojection_median_before_px']*.25
    assert np.array_equal(after[0]['world_to_camera'],estimated[0]['world_to_camera'])
    assert np.array_equal(after[1]['world_to_camera'],estimated[1]['world_to_camera'])
    assert diagnostic['all_previously_supported_tracks_retained']
    assert np.mean(np.linalg.norm(refined-truth,axis=1))<np.mean(np.linalg.norm(noisy-truth,axis=1))


def test_bundle_adjustment_honors_cancellation():
    scene=truth_scene()
    def cancelled():raise reconstruction.ReconstructionCancelled('cancelled')
    with pytest.raises(reconstruction.ReconstructionCancelled):
        bundle_adjust(scene['points'],scene['cameras'],scene['observations'],{0,1},check_cancel=cancelled)


def test_hybrid_alignment_only_adds_supported_inferred_points_and_preserves_observed():
    scene=truth_scene();original=copy.deepcopy(scene);prediction=learned_scene(scene)
    output,diagnostic=align_visible_completion(scene,prediction)
    assert diagnostic['status']=='completed_visible_only'
    assert diagnostic['added_points']>=12
    assert diagnostic['added_points']<len(prediction['points'])
    assert np.array_equal(output['points'][:len(scene['points'])],original['points'])
    assert output['cameras']==original['cameras']
    assert output['observations'][:len(scene['points'])]==original['observations']
    assert output['sources'][:len(scene['points'])]==original['sources']
    assert set(output['sources'][len(scene['points']):])=={'inferred'}
    assert all(len(t)>=2 for t in output['observations'][len(scene['points']):])
    assert np.array_equal(scene['points'],original['points'])
    assert diagnostic['unseen_surfaces_recovered'] is False


def test_hybrid_rejects_inconsistent_alignment_without_replacing_scene():
    scene=truth_scene();prediction=learned_scene(scene)
    rng=np.random.default_rng(33)
    for row in prediction['alignment_samples']:row['point']=rng.normal(0,10,3).tolist()
    output,diagnostic=align_visible_completion(scene,prediction)
    assert output is scene
    assert diagnostic['status']=='rejected' and diagnostic['added_points']==0


def test_structured_duplicate_and_featureless_failure(tmp_path):
    from PIL import Image
    one=tmp_path/'a.png';two=tmp_path/'b.png'
    Image.new('RGB',(80,80),'white').save(one);Image.new('RGB',(80,80),'black').save(two)
    with pytest.raises(reconstruction.ReconstructionError) as failure:
        reconstruction.build_sparse_scene([str(one),str(one)])
    assert failure.value.code=='duplicate_views'
    assert failure.value.diagnostics['duplicate_view_indices']==[1]
    with pytest.raises(reconstruction.ReconstructionError) as failure:
        reconstruction.build_sparse_scene([str(one),str(two)])
    assert failure.value.code=='insufficient_features'
    assert failure.value.diagnostics['external_pose_or_pointcloud_used'] is False


def test_auto_unavailable_completion_preserves_sfm_and_records_reason(monkeypatch):
    scene=truth_scene()
    monkeypatch.setattr(reconstruction,'build_sparse_scene',lambda *args:copy.deepcopy(scene))
    monkeypatch.setattr(reconstruction,'_learned_geometry_availability',lambda cfg:{'available':False,'requires_download':True,'reason':'not_authorized'})
    output=reconstruction.initialize_geometry(['0.png','1.png'],{'sparse_completion':'auto'})
    assert output['metadata']['sparse_completion']['status']=='unavailable'
    assert np.array_equal(output['points'],scene['points'])
    assert output['metadata']['geometry_backend_applied']=='sfm'


def test_auto_partial_registration_is_a_completion_trigger(monkeypatch):
    scene=truth_scene()
    monkeypatch.setattr(reconstruction,'build_sparse_scene',lambda *args:copy.deepcopy(scene))
    monkeypatch.setattr(reconstruction,'_learned_geometry_availability',lambda cfg:{'available':False,'reason':'missing'})
    output=reconstruction.initialize_geometry(['0.png','1.png','2.png'],{'sparse_completion':'auto'})
    assert 'partial_registration' in output['metadata']['sparse_completion']['triggers']
    assert output['metadata']['sparse_completion']['status']=='unavailable'


def test_auto_two_view_completion_actually_invokes_provider_and_keeps_sfm(tmp_path,monkeypatch):
    from PIL import Image
    scene=truth_scene();paths=[]
    for i,image in enumerate(scene['images']):
        path=tmp_path/f'{i}.png';Image.fromarray(image).save(path);paths.append(str(path))
    prediction=learned_scene(scene)
    # The standalone filtering test also contains an intentionally out-of-image
    # sample. The provider contract rejects that before reaching this hook.
    for key in ('points','colors','confidence','sources','observations'):prediction[key]=prediction[key][:-1]
    calls=[]
    def infer(paths,cfg,*args):
        calls.append(cfg)
        assert cfg['geometry_alignment_observations']==scene['observations']
        return copy.deepcopy(prediction)
    monkeypatch.setattr(reconstruction,'build_sparse_scene',lambda *args:copy.deepcopy(scene))
    monkeypatch.setitem(sys.modules,'backend.learned_geometry',SimpleNamespace(
        geometry_availability=lambda cfg:{'available':True,'requires_download':False},initialize_learned_geometry=infer))
    output=reconstruction.initialize_geometry(paths,{'sparse_completion':'auto'})
    assert len(calls)==1
    assert output['metadata']['sparse_completion']['status']=='completed_visible_only'
    assert np.array_equal(output['points'][:len(scene['points'])],scene['points'])
    assert output['cameras']==scene['cameras']


def test_router_failure_fallback_requires_permission_and_preserves_failure_provenance(monkeypatch):
    def fail(*args):raise reconstruction.ReconstructionError('few matches','insufficient_overlap',{'pairs':[]})
    monkeypatch.setattr(reconstruction,'build_sparse_scene',fail)
    scene=truth_scene();scene['sources']=['inferred']*len(scene['points']);scene['confidence'][:]=.25
    calls=[]
    module=SimpleNamespace(geometry_availability=lambda cfg:{'available':True,'requires_download':True,'reason':'download'},
        initialize_learned_geometry=lambda *args:(calls.append(True) or copy.deepcopy(scene)))
    monkeypatch.setitem(sys.modules,'backend.learned_geometry',module)
    with pytest.raises(reconstruction.ReconstructionError) as failure:
        reconstruction.initialize_geometry(['0.png','1.png'],{'allow_geometry_download':False})
    assert not calls
    assert failure.value.diagnostics['learned_fallback']['reason']=='geometry_model_download_not_authorized'
    output=reconstruction.initialize_geometry(['0.png','1.png'],{'allow_geometry_download':True})
    assert calls==[True]
    assert output['metadata']['sfm_failure']['code']=='insufficient_overlap'
    assert set(output['sources'])=={'inferred'}
    with pytest.raises(reconstruction.ReconstructionError):
        reconstruction.initialize_geometry(['0.png','1.png'],{'sparse_completion':'none','allow_geometry_download':True})
    assert len(calls)==1


def test_explicit_learned_geometry_rejects_observed_provenance(monkeypatch):
    scene=truth_scene();scene['confidence'][:]=.25
    monkeypatch.setitem(sys.modules,'backend.learned_geometry',SimpleNamespace(
        geometry_availability=lambda cfg:{'available':True,'requires_download':False},initialize_learned_geometry=lambda *args:scene))
    with pytest.raises(reconstruction.ReconstructionError,match='inferred'):
        reconstruction.initialize_geometry(['0.png','1.png'],{'geometry_backend':'dust3r'})


def test_sparse_training_changes_constraints_and_balances_view_sampling():
    scene=truth_scene();points=scene['points'][:2]
    common={'device':'cpu','iterations':9,'max_gaussians':2,'image_size':16}
    dense=reconstruction.train_gaussians(points,scene['colors'][:2],scene['cameras'],scene['images'],{**common,'sparse':False})
    sparse=reconstruction.train_gaussians(points,scene['colors'][:2],scene['cameras'],scene['images'],{**common,'sparse':True})
    a=dense['metrics']['effective_regularization'];b=sparse['metrics']['effective_regularization']
    assert b['position_anchor_weight']>a['position_anchor_weight']
    assert b['position_learning_rate_extent_factor']<a['position_learning_rate_extent_factor']
    assert b['scale_anchor_weight']>0
    visits=list(b['view_visit_counts'].values());assert sum(visits)==9 and max(visits)-min(visits)<=1
    extent=max(float(np.linalg.norm(np.std(points,axis=0))),.05)
    learned=np.asarray([g['position'] for g in sparse['gaussians']])
    assert np.max(np.abs(learned-points))<=extent*b['position_bound_extent_fraction']+1e-6


def test_sparse_preview_budget_retains_observed_support_among_inferred_points():
    scene=truth_scene();points=scene['points']
    output=reconstruction.train_gaussians(points,scene['colors'],scene['cameras'],scene['images'],
        {'device':'cpu','iterations':1,'max_gaussians':8,'image_size':16,'sparse':True,
         'initial_sources':['observed']*6+['inferred']*(len(points)-6)})
    assert sum(i<6 for i in output['selected_indices'])>=4
    assert output['metrics']['effective_regularization']['selected_observed_points']>=4
