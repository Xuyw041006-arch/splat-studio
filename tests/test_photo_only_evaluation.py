"""Independent checks for held-out registration and metric aggregation."""
import numpy as np
import pytest
import cv2
from benchmarks.evaluate_photo_only import aggregate_rows, category_columns, solve_test_pose, validate_photo_split, file_hash

def test_renamed_held_out_photo_cannot_enter_training(tmp_path):
    photo=tmp_path/'test.jpg';photo.write_bytes(b'photo identity')
    digest=file_hash(photo)
    project={'images':[{'name':'0000.jpg','original_name':'test.jpg','capture':{'source_sha256':digest}}]}
    protocol={'train_full':['0000.jpg'],'original_full':['test.jpg'],'evaluation_names':['test.jpg'],
              'source_hashes':{'test.jpg':digest}}
    with pytest.raises(ValueError,match='held-out original'):
        validate_photo_split(project,protocol,'full',tmp_path)

def test_native_photo_source_hash_is_checked(tmp_path):
    photo=tmp_path/'train.jpg';photo.write_bytes(b'original image')
    digest=file_hash(photo)
    project={'images':[{'name':'0000.jpg','original_name':'train.jpg','capture':{'source_sha256':digest}}]}
    protocol={'train_full':['0000.jpg'],'original_full':['train.jpg'],'evaluation_names':['test.jpg'],
              'source_hashes':{'train.jpg':digest}}
    assert validate_photo_split(project,protocol,'full',tmp_path)['source_hashes_verified']
    photo.write_bytes(b'changed image')
    with pytest.raises(ValueError,match='source identity changed'):
        validate_photo_split(project,protocol,'full',tmp_path)

def test_missing_semantic_category_counts_as_zero():
    rgb={'psnr_db':30.,'ssim':.9}
    rows=[{'rgb':rgb,'priority_rgb':rgb,'objects':[
        {'label':'bear','intersection':80,'union':100,'iou':.8,'boundary_iou':.5,'rgb':rgb},
        {'label':'mug','intersection':0,'union':40,'iou':0.,'boundary_iou':0.,'rgb':rgb}]}]
    result=aggregate_rows(rows)
    assert result['miou']==pytest.approx(.4)
    assert result['classes']['mug']['iou']==0
    assert result['overall_psnr_db']==30

def test_macro_iou_sums_pixels_per_class_before_averaging():
    rgb={'psnr_db':20.,'ssim':.8}
    rows=[{'rgb':rgb,'priority_rgb':rgb,'objects':[
        {'label':'bear','intersection':a,'union':b,'iou':a/b,'boundary_iou':.2,'rgb':rgb}]}
        for a,b in ((1,2),(9,90))]
    assert aggregate_rows(rows)['miou']==pytest.approx(10/92)

def test_class_lookup_includes_all_region_tracks_without_gt_selection():
    catalog={'classes':['r1','r2','r3'],'class_metadata':[
        {'display_label':'bear','source_labels':['bear']},
        {'display_label':'bear','source_labels':['bear']},
        {'display_label':'scene objects','source_labels':['scene objects']}]}
    assert category_columns(catalog,['bear','mug'])=={'bear':[0,1],'mug':[]}

def test_pnp_recovers_pose_with_outliers_and_different_focal():
    rng=np.random.default_rng(10)
    points=rng.uniform([-2,-1.4,4],[2,1.4,9],(220,3))
    k=np.array([[900.,0,494],[0,890,365],[0,0,1]])
    r=np.array([.05,-.09,.02]);t=np.array([.1,.03,.2])
    pixels=cv2.projectPoints(points,r,t,k,None)[0].reshape(-1,2)
    pixels+=rng.normal(0,.3,pixels.shape)
    pixels[:35]=rng.uniform([0,0],[988,730],(35,2))
    initial=k.copy();initial[0,0]*=.9;initial[1,1]*=.9
    recovered,calibration,audit=solve_test_pose(points,pixels,initial,988,730)
    assert audit['inliers']>=175
    assert audit['median_reprojection_pixels']<1
    np.testing.assert_allclose(recovered[:3,:3],cv2.Rodrigues(r)[0],atol=.003)
    np.testing.assert_allclose(recovered[:3,3],t,atol=.02)
    assert calibration[0,0]==pytest.approx(900,rel=.005)

def test_registration_rejects_too_few_correspondences():
    with pytest.raises(ValueError,match='30 distinct'):
        solve_test_pose(np.ones((5,3)),np.ones((5,2)),np.eye(3),100,100)
