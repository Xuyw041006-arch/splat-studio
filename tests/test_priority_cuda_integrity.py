"""CPU checks for the CUDA priority adapter's actual mask application contract."""
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'benchmarks'/f'{name}.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def test_priority_mask_keeps_complete_camera_filename(tmp_path):
    adapter=load('run_cuda_priority_ablation')
    scene=tmp_path/'scene';(scene/'images').mkdir(parents=True)
    Image.new('RGB',(4,4),'white').save(scene/'images/frame_00003.jpg')
    Image.new('L',(4,4),255).save(tmp_path/'mask.png')
    source=tmp_path/'masks.json'
    source.write_text(json.dumps({'ground_truth_used':False,'masks':[
        {'image_name':'frame_00003.jpg','label':'coffee mug','mask_path':'mask.png'}]}))
    result=adapter.prepare_masks(source,scene,'images',{'train':['frame_00003.jpg'],'test':['heldout.jpg']},tmp_path/'priority',['coffee mug'])
    assert (tmp_path/'priority/frame_00003.jpg.png').exists()
    assert not (tmp_path/'priority/frame_00003.png').exists()
    assert result['mask_filename_rule']=='exact_camera_image_name_plus_dot_png'


def hook(tmp_path,expected):
    import torch
    adapter=load('run_cuda_priority_ablation')
    scope={'os':os,'torch':torch,'SPLAT_BENCHMARK_PRIORITY_MASK_DIR':str(tmp_path),
        'SPLAT_BENCHMARK_PRIORITY_IMAGE_NAMES':expected}
    exec(compile(adapter.PRIORITY_HOOK,'priority_hook_test.py','exec'),scope)
    return scope


def test_nonempty_priority_pixels_actually_change_loss_and_are_counted(tmp_path):
    import torch
    mask=np.zeros((4,4),np.uint8);mask[:,:2]=255
    Image.fromarray(mask).save(tmp_path/'frame.jpg.png')
    scope=hook(tmp_path,['frame.jpg'])
    image=torch.zeros((3,4,4),requires_grad=True);image=image+torch.tensor((mask>0).astype(np.float32))[None]
    target=torch.zeros_like(image)
    value=scope['splat_weighted_l1'](image,target,'frame.jpg')
    assert value.item()==pytest.approx(.75)
    assert torch.abs(image-target).mean().item()==pytest.approx(.5)
    value.backward()
    scope['splat_weighted_l1'](target,target,'background.jpg')
    audit=scope['splat_priority_report']()
    assert audit['calls']=={'total':2,'weighted':1,'background_only':1}
    assert audit['loaded_nonempty_image_names']==['frame.jpg']


def test_old_stem_only_filename_is_rejected_instead_of_all_one_fallback(tmp_path):
    import torch
    Image.new('L',(4,4),255).save(tmp_path/'frame.png')
    scope=hook(tmp_path,['frame.jpg']);image=torch.ones(3,4,4)
    with pytest.raises(RuntimeError,match='exact camera image_name'):
        scope['splat_weighted_l1'](image,image,'frame.jpg')


def test_zero_hits_or_unvisited_priority_view_cannot_publish_success(tmp_path):
    scope=hook(tmp_path,['frame.jpg'])
    with pytest.raises(RuntimeError,match='not applied'):
        scope['splat_priority_report']()


def test_worker_compiles_and_checks_priority_hits_before_timing_publication():
    benchmark=load('run_cuda_benchmark');adapter=load('run_cuda_priority_ablation')
    source=adapter.build_worker(benchmark.WORKER)
    compile(source,'priority_worker.py','exec')
    assert source.index('timing["priority_loss_application"] = train.splat_priority_report()')<source.index('(output/"training_timing.json").write_text')
    assert 'SPLAT_BENCHMARK_PRIORITY_IMAGE_NAMES' in source
