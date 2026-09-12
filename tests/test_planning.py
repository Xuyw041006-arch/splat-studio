import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.planning import make_plan
import pytest


def analysis_for(n,unique=None,matches=100):
    return {'image_count':n,'unique_image_count':n if unique is None else unique,'connected_ratio':1.,'max_pair_inliers':matches,'warnings':[]}


def learned_hardware():
    return {'device':'mps','original_3dgs':False,'learned_geometry':{'available':True,'requires_download':False}}

def test_fine_cuda_budget_is_22000_without_changing_preview_budget():
    analysis={'image_count':30,'connected_ratio':1.,'max_pair_inliers':100,'warnings':[]}
    plan=make_plan(analysis,{'mode':'fine'},{'device':'cuda','original_3dgs':True})
    assert plan['settings']['iterations']==22000
    assert plan['settings']['resolution']==1
    assert plan['settings']['preview_steps']==120

def test_geometry_and_span_drive_sparse_strategy_and_estimates_are_ranges():
    hardware={'device':'cpu','original_3dgs':False}
    analysis={'image_count':30,'connected_ratio':1.,'max_pair_inliers':100,'warnings':[]}
    assert make_plan(analysis,{'mode':'fast','view_span':360},hardware)['sparse'] is False
    assert make_plan(analysis,{'mode':'fast','view_span':30},hardware)['sparse'] is True
    times=[make_plan(analysis,{'mode':mode},hardware)['estimated_seconds'][1] for mode in ('fast','balanced','fine')]
    assert times[0]<times[1]<times[2]


def test_none_disables_fallback_planning_even_with_ready_model_and_bad_matching():
    plan=make_plan(analysis_for(2,matches=0),{'geometry_backend':'auto','sparse_completion':'none'},learned_hardware())
    assert not plan['can_reconstruct']
    assert plan['automatic_completion_status']=='not_requested'
    assert not any('自动补全将使用' in warning for warning in plan['warnings'])


@pytest.mark.parametrize('backend,matches,expected',[('dust3r',100,False),('auto',100,True),('auto',0,False)])
def test_user_calibration_never_plans_unusable_learned_path(backend,matches,expected):
    config={'geometry_backend':backend,'intrinsics_original':[[[72,0,40],[0,72,20],[0,0,1]]]*2,
            'intrinsics_sources':['user_calibrated']*2}
    plan=make_plan(analysis_for(2,matches=matches),config,learned_hardware())
    assert plan['can_reconstruct'] is expected
    assert plan['automatic_completion_status']=='unavailable'
    assert 'learned_calibration_unsupported' in plan['learned_geometry_blockers']


def test_approximate_exif_intrinsics_allow_inferred_focal_estimation():
    plan=make_plan(analysis_for(2),{'geometry_backend':'dust3r','intrinsics_original':[[[72,0,40],[0,72,20],[0,0,1]]]*2,
        'intrinsics_sources':['heuristic','exif_35mm_approximate']},learned_hardware())
    assert plan['can_reconstruct'] and plan['automatic_completion_status']=='ready'


def test_any_duplicate_blocks_dust3r_but_does_not_prevent_valid_sfm_attempt():
    analysis=analysis_for(3,unique=2)
    direct=make_plan(analysis,{'geometry_backend':'dust3r'},learned_hardware())
    assert not direct['can_reconstruct']
    assert 'learned_duplicate_views' in direct['learned_geometry_blockers']
    automatic=make_plan(analysis,{'geometry_backend':'auto'},learned_hardware())
    assert automatic['can_reconstruct'] and automatic['automatic_completion_status']=='unavailable'


@pytest.mark.parametrize('count,expected',[(1,False),(2,True),(12,True),(13,False)])
def test_learned_view_cap_matches_initializer(count,expected):
    plan=make_plan(analysis_for(count),{'geometry_backend':'dust3r'},learned_hardware())
    assert plan['can_reconstruct'] is expected
    assert plan['learned_geometry_input_eligible'] is expected


def test_photo_geometry_capacity_and_colmap_route_are_distinct():
    a=analysis_for(81)
    assert not make_plan(a,{'geometry_backend':'sfm'},learned_hardware())['can_reconstruct']
    cuda={'device':'cuda','original_3dgs':True,'colmap':True}
    standard=make_plan(a,{'geometry_backend':'auto','sparse_completion':'none'},cuda)
    assert standard['can_reconstruct'] and standard['geometry_initialization_path']=='colmap_photo_initialization'
    sparse=make_plan(a,{'geometry_backend':'auto','view_span':30},cuda)
    assert not sparse['can_reconstruct'] and sparse['geometry_initialization_path']=='shared_photo_initialization'


def test_download_permission_is_required_even_if_capability_claims_available():
    hardware=learned_hardware();hardware['learned_geometry']['requires_download']=True
    denied=make_plan(analysis_for(2,matches=0),{'geometry_backend':'dust3r'},hardware)
    assert not denied['can_reconstruct']
    allowed=make_plan(analysis_for(2,matches=0),{'geometry_backend':'dust3r','allow_geometry_download':True},hardware)
    assert allowed['can_reconstruct']
