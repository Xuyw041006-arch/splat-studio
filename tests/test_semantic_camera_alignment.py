import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.upstream import registered_semantic_cameras


def test_registered_intrinsics_keep_principal_point_and_reject_changed_mask_grid(tmp_path):
    reader_path=Path(__file__).resolve().parents[1]/'vendor/gaussian-splatting/utils/read_write_model.py'
    spec=importlib.util.spec_from_file_location('test_colmap_reader',reader_path)
    reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    original=tmp_path/'original';current=tmp_path/'dataset/sparse/0'
    original.mkdir();current.mkdir(parents=True)
    camera=reader.Camera(1,'PINHOLE',4,4,np.array([4.,5.,1.7,1.8]))
    image=reader.Image(1,np.array([1.,0.,0.,0.]),np.zeros(3),1,'view.png',np.empty((0,2)),np.empty(0,dtype=np.int64))
    for folder in (original,current):
        reader.write_cameras_binary({1:camera},folder/'cameras.bin')
        reader.write_images_binary({1:image},folder/'images.bin')
    result=registered_semantic_cameras(tmp_path/'dataset',original,reader_path)
    assert result[0]['intrinsics']['cx']==1.7
    assert result[0]['intrinsics']['cy']==1.8
    assert result[0]['original_registered_pixel_grid_equal'] is True
    assert result[0]['mask_pixel_coordinates_verified'] is False
    assert result[0]['rgb_symmetric_projection_matches'] is False
    centered=reader.Camera(1,'PINHOLE',4,4,np.array([4.,5.,2.,2.]))
    for folder in (original,current):
        reader.write_cameras_binary({1:centered},folder/'cameras.bin')
    result=registered_semantic_cameras(tmp_path/'dataset',original,reader_path)
    assert result[0]['mask_pixel_coordinates_verified'] is True
    changed=reader.Camera(1,'PINHOLE',4,4,np.array([4.,5.,1.8,1.8]))
    reader.write_cameras_binary({1:changed},current/'cameras.bin')
    result=registered_semantic_cameras(tmp_path/'dataset',original,reader_path)
    assert result[0]['mask_pixel_coordinates_verified'] is False


def test_priority_coordinate_gate_checks_full_grid_not_symmetric_intrinsics(tmp_path):
    from PIL import Image
    from backend.upstream import _validate_priority_mask_coordinates
    Image.new('L',(4,4),255).save(tmp_path/'view.png.png')
    camera={'image_name':'view.png','width':4,'height':4,
            'original_registered_pixel_grid_equal':True,
            'mask_pixel_coordinates_verified':False,'rgb_symmetric_projection_matches':False}
    # Off-center intrinsics affect semantic projection but do not move the RGB
    # target pixels used by the photometric mask loss when the grid is unchanged.
    assert _validate_priority_mask_coordinates([camera],tmp_path)==['view.png']
    camera['original_registered_pixel_grid_equal']=False
    with pytest.raises(ValueError,match='像素网格不一致'):
        _validate_priority_mask_coordinates([camera],tmp_path)
    camera['original_registered_pixel_grid_equal']=True
    Image.new('L',(2,4),255).save(tmp_path/'view.png.png')
    with pytest.raises(ValueError,match='完整图像尺寸'):
        _validate_priority_mask_coordinates([camera],tmp_path)
    (tmp_path/'view.png.png').rename(tmp_path/'unregistered.png.png')
    with pytest.raises(ValueError,match='没有对应'):
        _validate_priority_mask_coordinates([camera],tmp_path)


@pytest.mark.parametrize('changed_grid',[False,True])
def test_priority_reconstruction_validates_real_colmap_metadata_before_rgb_training(tmp_path,monkeypatch,changed_grid):
    import shutil
    import torch
    from PIL import Image
    from backend import upstream
    repo=Path(__file__).resolve().parents[1]/'vendor/gaussian-splatting'
    spec=importlib.util.spec_from_file_location('priority_colmap_reader',repo/'utils/read_write_model.py')
    reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    original=reader.Camera(1,'PINHOLE',4,4,np.array([4.,5.,2.,2.]))
    current=reader.Camera(1,'PINHOLE',4,4,np.array([4.,5.,1.8 if changed_grid else 2.,2.]))
    view=reader.Image(1,np.array([1.,0.,0.,0.]),np.zeros(3),1,'view.jpg',np.empty((0,2)),np.empty(0,dtype=np.int64))
    image=tmp_path/'view.jpg';Image.new('RGB',(4,4)).save(image)
    masks=tmp_path/'priority';masks.mkdir();Image.new('L',(4,4),255).save(masks/'view.jpg.png')
    output=tmp_path/'run';calls=[]
    class TrainingReached(Exception):pass
    def fake_run(argv,cwd,log,progress,cancelled,stage,iteration_total=None):
        calls.append(stage)
        if argv[0]!='colmap':raise TrainingReached()
        if argv[1]=='mapper':
            folder=Path(argv[argv.index('--output_path')+1])/'0';folder.mkdir()
            reader.write_cameras_binary({1:original},folder/'cameras.bin')
            reader.write_images_binary({1:view},folder/'images.bin')
        elif argv[1]=='image_undistorter':
            folder=Path(argv[argv.index('--output_path')+1])/'sparse';folder.mkdir()
            reader.write_cameras_binary({1:current},folder/'cameras.bin')
            reader.write_images_binary({1:view},folder/'images.bin')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(shutil,'which',lambda name:'/mock/colmap')
    monkeypatch.setattr(upstream,'run_logged',fake_run)
    monkeypatch.setenv('SPLAT_3DGS_REPO',str(repo))
    monkeypatch.setenv('SPLAT_PRIORITY_MASK_DIR','')
    expected=RuntimeError if changed_grid else TrainingReached
    match='优先物品训练已停止.*像素网格不一致' if changed_grid else None
    with pytest.raises(expected,match=match):
        upstream.reconstruct_upstream([image],output,{'mode':'fast','priority_masks':True,'priority_masks_dir':str(masks)},lambda *_:None,lambda:False)
    assert ('原始 3DGS CUDA 优化' in calls) is (not changed_grid)
    assert (output/'train_priority.py').exists() is (not changed_grid)
