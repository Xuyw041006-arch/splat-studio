import copy
import io
import numpy as np
import pytest
from PIL import Image, ImageOps
from backend.capture import photo_metadata, validate_calibration, geometry_config, capture_fingerprint
from backend.planning import largest_component, make_plan


def manifest():
    image=Image.new('RGB',(80,40))
    return {'images':[{'name':f'{i:04d}.jpg','width':80,'height':40,
        'capture':photo_metadata(image,image,bytes([i]))} for i in range(2)]}


def payload(m):
    return {'coordinate_space':'uploaded_pixels','cameras':[
        {'image_name':i['name'],'width':80,'height':40,'K':i['capture']['K'],
         'source_sha256':i['capture']['source_sha256']} for i in m['images']]}


def test_upright_exif_focal_grid_and_no_private_metadata():
    image=Image.new('RGB',(80,40)); exif=Image.Exif();exif[274]=6;exif[41989]=50;exif[315]='private author'
    stream=io.BytesIO();image.save(stream,format='JPEG',exif=exif)
    with Image.open(io.BytesIO(stream.getvalue())) as opened:
        upright=ImageOps.exif_transpose(opened)
        record=photo_metadata(opened,upright,stream.getvalue())
    assert upright.size==(40,80)
    assert record['K'][0][2]==20 and record['K'][1][2]==40
    assert record['intrinsics_source']=='exif_35mm_approximate'
    assert 'private author' not in str(record)
    assert record['K'][0][0]==pytest.approx(50*np.hypot(40,80)/np.hypot(36,24))


@pytest.mark.parametrize('mutate',[
    lambda p:p.update(coordinate_space='raw_sensor'),
    lambda p:p['cameras'][0].update(width=40),
    lambda p:p['cameras'][0].update(source_sha256='unbound'),
    lambda p:p['cameras'][0].update(image_name='0001.jpg'),
    lambda p:p['cameras'][0].update(w2c=np.eye(4).tolist()),
    lambda p:p['cameras'][0].update(K=[[float('nan'),0,40],[0,72,20],[0,0,1]]),
    lambda p:p['cameras'][0].update(K=[[72,1,40],[0,72,20],[0,0,1]]),
])
def test_calibration_rejects_wrong_image_or_pixel_grid(mutate):
    m=manifest(); p=payload(m);mutate(p)
    with pytest.raises(ValueError):validate_calibration(p,m['images'])


def test_calibration_changes_fingerprint_and_preserves_grid_contract():
    m=manifest(); before=capture_fingerprint(m);p=payload(copy.deepcopy(m))
    p['cameras'][0]['K'][0][0]=100
    values=validate_calibration(p,m['images'])
    for i in m['images']:i['calibrated_K']=values[i['name']]
    cfg=geometry_config(m,{'device':'cpu'})
    assert capture_fingerprint(m)!=before
    assert cfg['original_sizes']==[[80,40],[80,40]]
    assert cfg['intrinsics_original'][0][0][0]==100
    assert cfg['intrinsics_sources']==['user_calibrated']*2
    assert cfg['geometry_device']=='cpu'


@pytest.mark.parametrize('missing',[None,'','not-a-sha'])
def test_calibration_never_accepts_missing_or_invalid_identity_even_when_both_match(missing):
    m=manifest();m['images'][0]['capture']['source_sha256']=missing;p=payload(m)
    with pytest.raises(ValueError,match='source_sha256'):
        validate_calibration(p,m['images'])


def test_calibration_rejects_non_string_name_without_internal_type_error():
    m=manifest();p=payload(m);p['cameras'][0]['image_name']=['0000.jpg']
    with pytest.raises(ValueError,match='image_name'):
        validate_calibration(p,m['images'])


def test_malformed_orientation_metadata_does_not_reject_otherwise_valid_pixels():
    image=Image.new('RGB',(80,40));image.getexif()[274]='invalid'
    metadata=photo_metadata(image,image,b'pixels')
    assert metadata['exif_orientation']==1
    assert metadata['K'][0][2]==40


def test_graph_uses_largest_component_not_first_photo():
    edges=[{'i':i,'j':i+1,'valid':True} for i in (1,2,3)]
    assert largest_component(range(5),edges)==[1,2,3,4]


def test_plan_does_not_allow_learned_to_rescue_duplicate_images():
    a={'image_count':2,'unique_image_count':1,'connected_ratio':.5,'max_pair_inliers':0,'warnings':[]}
    h={'device':'mps','original_3dgs':False,'learned_geometry':{'available':True}}
    assert not make_plan(a,{'geometry_backend':'dust3r'},h)['can_reconstruct']
    a['unique_image_count']=2
    assert make_plan(a,{'geometry_backend':'auto'},h)['can_reconstruct']
    assert not make_plan(a,{'geometry_backend':'auto','sparse_completion':'none'},h)['can_reconstruct']
    assert not make_plan(a,{'geometry_backend':'sfm'},h)['can_reconstruct']


def test_user_calibration_forces_shared_route_and_offcenter_cuda_is_rejected():
    from backend.upstream import use_shared_initialization
    cfg={'intrinsics_sources':['user_calibrated']*30,'intrinsics_original':[[[100,0,40],[0,100,20],[0,0,1]]]*30,'original_sizes':[[80,40]]*30}
    analysis={'image_count':30,'unique_image_count':30,'connected_ratio':1.,'max_pair_inliers':100,'warnings':[]}
    hardware={'device':'cuda','original_3dgs':True,'colmap':True}
    assert use_shared_initialization(30,cfg,True)
    assert make_plan(analysis,cfg,hardware)['geometry_initialization_path']=='shared_photo_initialization'
    cfg['intrinsics_original']=copy.deepcopy(cfg['intrinsics_original']);cfg['intrinsics_original'][0][0][2]=39
    assert make_plan(analysis,cfg,hardware)['can_reconstruct'] is False
