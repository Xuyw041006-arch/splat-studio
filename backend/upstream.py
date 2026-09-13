"""Run the pinned GRAPHDECO trainer without altering the upstream checkout."""
from __future__ import annotations
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from .planning import MODES
from .gaussian_io import read_ply, write_scene
from .colmap_export import export_initialization_dataset, file_sha256
from .sparse_cuda import build_sparse_depth_anchors, patch_training_source, sparse_densification_settings

UPSTREAM_COMMIT='54c035f7834b564019656c3e3fcc3646292f727d'

def registered_semantic_cameras(dataset, original_model, reader_path):
    """Use registered intrinsics and verify original masks share their pixel grid.

    Iterative semantics must not silently project distorted-input masks onto an
    undistorted camera. Only unchanged pinhole calibration is accepted here.
    """
    import importlib.util
    import numpy as np
    spec=importlib.util.spec_from_file_location('_splat_semantic_colmap_reader',reader_path)
    reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    dataset=Path(dataset);original_model=Path(original_model)
    current_cameras=reader.read_cameras_binary(dataset/'sparse/0/cameras.bin')
    current_images=reader.read_images_binary(dataset/'sparse/0/images.bin')
    original_cameras=reader.read_cameras_binary(original_model/'cameras.bin')
    original_images={i.name:i for i in reader.read_images_binary(original_model/'images.bin').values()}
    def intrinsics(camera):
        if camera.model=='PINHOLE':fx,fy,cx,cy=camera.params
        elif camera.model=='SIMPLE_PINHOLE':f,cx,cy=camera.params;fx=fy=f
        else:raise ValueError('语义掩码与注册相机尚未完成畸变坐标转换')
        return np.array([fx,fy,cx,cy],dtype=float)
    cameras=[]
    for image in current_images.values():
        camera=current_cameras[image.camera_id];k=intrinsics(camera)
        original_image=original_images.get(image.name)
        previous=original_cameras[original_image.camera_id] if original_image is not None else None
        same_grid=bool(previous is not None and previous.model in {'PINHOLE','SIMPLE_PINHOLE'} and
                      (previous.width,previous.height)==(camera.width,camera.height) and
                      np.allclose(intrinsics(previous),k,rtol=1e-7,atol=1e-6))
        # The pinned RGB trainer uses symmetric FoV and ignores COLMAP cx/cy.
        # Off-center calibration cannot silently be used for a different
        # semantic projection on geometry optimized with that RGB renderer.
        centered=bool(np.allclose(k[2:],[camera.width/2,camera.height/2],rtol=0,atol=1e-6))
        verified=same_grid and centered
        matrix=np.eye(4);matrix[:3,:3]=reader.qvec2rotmat(image.qvec);matrix[:3,3]=image.tvec
        cameras.append({'image_name':image.name,'width':int(camera.width),'height':int(camera.height),
            'intrinsics':dict(zip(('fx','fy','cx','cy'),k.tolist())), 'world_to_camera':matrix.tolist(),
            'mask_pixel_coordinates_verified':verified,
            'original_registered_pixel_grid_equal':same_grid,'rgb_symmetric_projection_matches':centered,
            'mask_coordinate_source':'original_vs_registered_pinhole_and_pinned_RGB_symmetric_projection'})
    return cameras

def _validate_priority_mask_coordinates(cameras, mask_dir):
    """Reject input-grid masks that cannot supervise registered RGB pixels.

    The priority loss compares rendered and target pixels, so it needs the same
    image grid; semantic center projection additionally requires centered RGB
    intrinsics. Normal trainer resolution scaling is applied only after this
    full-resolution check.
    """
    from PIL import Image
    directory=Path(mask_dir)
    if not directory.is_dir():raise ValueError('优先物品掩码目录不可用')
    checked=[]
    for camera in cameras:
        name=camera['image_name']
        path=directory/(name+'.png')
        if not path.is_file():continue
        if camera.get('original_registered_pixel_grid_equal') is not True:
            raise ValueError(f'{name} 的原图掩码与注册训练图像像素网格不一致；需先转换去畸变/裁剪坐标，不能只缩放掩码')
        with Image.open(path) as mask:
            if mask.size!=(camera['width'],camera['height']):
                raise ValueError(f'{name} 的优先物品掩码不是已验证的完整图像尺寸')
        checked.append(name)
    if not checked:raise ValueError('已注册训练相机中没有对应的优先物品掩码，无法执行物品加权训练')
    return checked

def run_logged(argv, cwd, log_path, progress, cancelled, stage, iteration_total=None):
    """Argv-only subprocess, streamed logs, cancellation even if subprocess is quiet."""
    lines=queue.Queue()
    env={**os.environ,'PYTHONUNBUFFERED':'1'}
    proc=subprocess.Popen([str(v) for v in argv],cwd=str(cwd),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                          text=True,encoding='utf-8',errors='replace',env=env)
    def read():
        for line in proc.stdout: lines.put(line)
        lines.put(None)
    threading.Thread(target=read,daemon=True).start()
    tail=[];last_progress=.3;finished_iterations=False
    with open(log_path,'a',encoding='utf-8') as log:
        log.write('\nCOMMAND: '+repr(argv)+'\n')
        try:
            done=False
            while not done or proc.poll() is None:
                if cancelled():
                    proc.terminate()
                    try: proc.wait(timeout=5)
                    except subprocess.TimeoutExpired: proc.kill();proc.wait()
                    raise RuntimeError('任务已取消')
                try: line=lines.get(timeout=0.2)
                except queue.Empty: continue
                if line is None: done=True;continue
                log.write(line);log.flush();tail=(tail+[line.strip()])[-12:]
                if iteration_total:
                    # Only the trainer's tqdm counter is training progress.
                    match=re.search(r'Training progress:.*?(\d+)\s*/\s*'+str(iteration_total)+r'\b',line)
                    if match and int(match.group(1))<=iteration_total:
                        current=int(match.group(1));last_progress=.3+.62*current/iteration_total
                        finished_iterations=current==iteration_total
                        progress(last_progress, '训练迭代完成，正在验证或保存' if finished_iterations else stage+' '+line.strip())
                    elif re.search(r'\bSaving (?:Gaussians|Checkpoint)\b',line):
                        progress(last_progress,'保存高斯模型或检查点 · 当前阶段无可靠 ETA')
                    elif re.search(r'\bEvaluating\b',line):
                        progress(last_progress,'验证渲染结果 · 当前阶段无可靠 ETA')
            if proc.wait()!=0: raise RuntimeError(stage+' 失败：\n'+'\n'.join(tail))
        finally:
            if proc.poll() is None: proc.terminate();proc.wait()

def use_shared_initialization(image_count, config, colmap_available):
    """Small sparse inputs use the same actual-photo geometry as the MPS path."""
    requested = (bool(config.get('sparse')) or config.get('geometry_backend', 'auto') in {'sfm', 'dust3r'}
                 or config.get('sparse_completion') == 'learned_visible'
                 or 'user_calibrated' in config.get('intrinsics_sources', []))
    return requested or 2 <= image_count <= 6 or (not colmap_available and image_count <= 80)


def reconstruct_upstream(image_paths, output_dir, config, progress, cancelled):
    import torch
    if config.get('completion') == 'learned':
        raise RuntimeError('CUDA 路径尚不支持外部背面补全 provider；请使用 sparse_completion 的可见区域初始化，不能将它视为未见表面恢复。')
    if not torch.cuda.is_available(): raise RuntimeError('原始 3DGS 需要 NVIDIA CUDA；当前设备不可用。')
    colmap_available = bool(shutil.which('colmap'))
    shared = use_shared_initialization(len(image_paths), config, colmap_available)
    if not shared and not colmap_available: raise RuntimeError('未找到 COLMAP；超过 80 张照片的标准 CUDA 流程需要安装 COLMAP。')
    root=Path(os.environ.get('SPLAT_3DGS_REPO',Path(__file__).resolve().parents[1]/'vendor/gaussian-splatting')).resolve()
    if not (root/'train.py').is_file(): raise RuntimeError('原始 3DGS 仓库缺失，请运行 scripts/setup_upstream.py')
    out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    dataset=out/'dataset'
    log=out/'training.log'
    def run(args,stage,total=None): run_logged(args,root,log,progress,cancelled,stage,total)
    exported = None; anchors = None; adapter = None; lineage = None
    if shared:
        from .reconstruction import initialize_geometry
        progress(.02, '仅从实际输入照片初始化相机与稀疏几何')
        initialized = initialize_geometry(image_paths, config,
            lambda value, message: progress(min(.24, max(.02, float(value) * .7)), message), cancelled)
        exported = export_initialization_dataset(initialized, image_paths, dataset, cancelled)
        model = Path(exported['model_path'])
        del initialized
        progress(.26, '建立输入观测支持的稀疏逆深度锚点')
        anchors = build_sparse_depth_anchors(exported, out/'sparse_depth')
        if float(config.get('sparse_depth_weight', .02)) > 0 and not anchors['metadata']['total_anchors']:
            raise RuntimeError('初始化没有有效的输入观测深度锚点，无法启用稀疏 CUDA 约束；请补充有重叠与位移的照片。')
    else:
        inputs=dataset/'input';inputs.mkdir(parents=True,exist_ok=True)
        for p in image_paths: shutil.copy2(p,inputs/Path(p).name)
        dist=dataset/'distorted';dist.mkdir(exist_ok=True);sparse=dist/'sparse';sparse.mkdir(exist_ok=True)
        progress(.02,'COLMAP 特征提取')
        run(['colmap','feature_extractor','--database_path',dist/'database.db','--image_path',inputs,
             '--ImageReader.camera_model','PINHOLE','--ImageReader.single_camera','0'],'COLMAP 特征提取')
        progress(.1,'COLMAP 跨视角匹配')
        run(['colmap','exhaustive_matcher','--database_path',dist/'database.db'],'COLMAP 特征匹配')
        progress(.17,'COLMAP 相机标定与稀疏几何')
        run(['colmap','mapper','--database_path',dist/'database.db','--image_path',inputs,'--output_path',sparse], 'COLMAP 相机注册')
        models=[p for p in sparse.iterdir() if p.is_dir() and (p/'images.bin').is_file()]
        if not models: raise RuntimeError('COLMAP 无法注册相机；请增加有重叠和位移的照片。')
        model=max(models,key=lambda p:(p/'images.bin').stat().st_size)
        run(['colmap','image_undistorter','--image_path',inputs,'--input_path',model,'--output_path',dataset,'--output_type','COLMAP'],'图像去畸变')
        dst=dataset/'sparse'/'0';dst.mkdir(parents=True,exist_ok=True)
        for p in (dataset/'sparse').iterdir():
            if p.is_file(): shutil.move(str(p),dst/p.name)
    settings=MODES[config.get('mode','balanced')];iterations=settings['iterations']
    train=root/'train.py'
    verified_cameras=exported['cameras'] if exported else None;priority_camera_names=[]
    if config.get('priority_masks'):
        # Fail before photometric training rather than weighting unrelated
        # pixels when COLMAP has changed the original mask coordinate system.
        try:
            if verified_cameras is None:
                verified_cameras=registered_semantic_cameras(dataset,model,root/'utils/read_write_model.py')
            priority_camera_names=_validate_priority_mask_coordinates(verified_cameras,config['priority_masks_dir'])
        except (OSError,ValueError,KeyError) as exc:
            raise RuntimeError('优先物品训练已停止：掩码像素坐标校验失败。'+str(exc)) from exc
    if shared or config.get('priority_masks'):
        if shared:
            from .gaussian_lineage import prepare_lineage_tracking
            lineage=prepare_lineage_tracking(exported['metadata']['point_provenance_path'],
                Path(exported['model_path'])/'points3D.ply',root,out)
        patched = out / ('train_sparse.py' if shared else 'train_priority.py')
        adapter = patch_training_source(train, patched,
            anchor_manifest=anchors['manifest_path'] if anchors else None,
            depth_weight=config.get('sparse_depth_weight', .02),
            priority_dir=config.get('priority_masks_dir') if config.get('priority_masks') else None,
            lineage_hook=lineage['hook_source'] if lineage else None,
            priority_detail=__import__('backend.priority_detail',fromlist=['detail_profile']).detail_profile(config.get('mode','balanced'),iterations)
                if config.get('priority_masks') and config.get('priority_refinement')=='detail' else None)
        train = patched
        (out/'trainer_adapter.json').write_text(json.dumps(adapter, ensure_ascii=False, indent=2)+'\n')
    progress(.3,'原始 3DGS CUDA 优化')
    runner=[sys.executable,'-c','import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); script=sys.argv.pop(1); sys.argv[0]=script; runpy.run_path(script,run_name="__main__")',str(root),str(train)]
    runner+=['-s',str(dataset),'-m',str(out/'model'),'--iterations',str(iterations),'-r',str(settings['resolution']),
             '--save_iterations',str(iterations),'--test_iterations',str(iterations),'--disable_viewer']
    densification = sparse_densification_settings(iterations) if shared else None
    if shared:
        runner+=['--position_lr_init','0.00008','--scaling_lr','0.0025']
        for name, value in densification.items(): runner += ['--'+name, str(value)]
    run(runner,'原始 3DGS CUDA 优化',iterations)
    return finalize_upstream_model(image_paths, out, config, root=root, source_model=model,
        exported=exported, anchors=anchors, adapter=adapter, lineage=lineage,
        verified_cameras=verified_cameras, priority_camera_names=priority_camera_names,
        progress=progress)


def finalize_upstream_model(image_paths, output_dir, config, *, root, source_model,
                            exported=None, anchors=None, adapter=None, lineage=None,
                            verified_cameras=None, priority_camera_names=(), progress=lambda *_:None):
    """Serialize an already-saved upstream model; never invoke geometry or RGB training.

    Separating this stage lets an explicitly audited recovery retain a completed
    PLY after a transport/capacity error, including its SH and priority evidence.
    The caller must establish the saved run's identity before recovery.
    """
    out=Path(output_dir).resolve();root=Path(root).resolve();model=Path(source_model).resolve()
    dataset=out/'dataset';shared=exported is not None
    iterations=MODES[config.get('mode','balanced')]['iterations']
    densification=sparse_densification_settings(iterations) if shared else None
    progress(.93,'读取原始模型并生成交互预览 · 当前阶段无可靠 ETA')
    ply=out/'model'/'point_cloud'/f'iteration_{iterations}'/'point_cloud.ply'
    from .scene_limits import scene_point_budget
    preview_limit=scene_point_budget(config.get('viewer_quality','full'))
    scene=read_ply(ply,max_points=preview_limit)
    if lineage:
        from .gaussian_lineage import apply_lineage_to_preview
        apply_lineage_to_preview(scene,ply)
    camera_path=out/'model'/'cameras.json'
    scene['metadata'].update({'backend':'original_3dgs','upstream_commit':UPSTREAM_COMMIT,'ply_path':str(ply),
                              'training_mode':config.get('mode'),'sparse_strategy':'shared_input_geometry_and_sparse_inverse_depth' if shared else 'standard_colmap',
                              'completion':'not_performed','viewer_quality':config.get('viewer_quality','full'),
                              'warning':'显示保留原模型全部球谐阶数；默认全量高斯，只有明确选择预览预算才抽样。完整模型保存在 PLY。'})
    if exported:
        scene['metadata'].update(initialization=exported['metadata'], sparse_depth=anchors['metadata'],
            trainer_adapter=adapter, sparse_densification=densification,
            lineage_tracking={key:value for key,value in lineage.items() if key!='hook_source'},
            sparse_completion=exported['metadata']['initialization'].get('sparse_completion', {}),
            training_input_manifest=exported['metadata']['input_manifest'])
    else:
        scene['metadata']['training_input_manifest']=[{'image_name':Path(p).name,'source_sha256':file_sha256(p)} for p in image_paths]
    if priority_camera_names:
        scene['metadata']['priority_mask_coordinate_validation']={'status':'verified_original_registered_pixel_grid',
            'image_names':priority_camera_names,'training_resolution_resize_only':True}
    detail_evidence=out/'priority-detail/evidence.json'
    if config.get('priority_masks') and config.get('priority_refinement')=='detail':
        if not detail_evidence.is_file():raise RuntimeError('重点细化没有生成验收证据，任务不能标记完成')
        scene['metadata']['priority_enhancement']={**json.loads(detail_evidence.read_text()),'labels':config['priority_objects'],
            'request':config.get('priority_request',''),'approval_sha256':config.get('priority_approval_sha256','')}
    if camera_path.exists():
        import numpy as np
        cameras=[]
        for c in json.loads(camera_path.read_text()):
            c2w=np.eye(4);c2w[:3,:3]=c['rotation'];c2w[:3,3]=c['position']
            name=next((Path(p).name for p in image_paths if Path(p).stem==c['img_name']),c['img_name'])
            cameras.append({'image_name':name,'width':c['width'],'height':c['height'],
                'intrinsics':{'fx':c['fx'],'fy':c['fy'],'cx':c['width']/2,'cy':c['height']/2},'world_to_camera':np.linalg.inv(c2w).tolist()})
        scene['cameras']=cameras
    try:
        scene['cameras']=verified_cameras if verified_cameras is not None else registered_semantic_cameras(dataset,model,root/'utils/read_write_model.py')
        scene['metadata']['registered_semantic_camera_source']='shared input-only initialization with verified original pixels' if shared else 'COLMAP sparse/0 with original mask-grid check'
    except (OSError,ValueError,KeyError) as exc:
        scene['metadata'].setdefault('warnings',[]).append('多轮语义坐标校验未完成：'+str(exc))
    scene_path=out/'scene.json';write_scene(scene,scene_path)
    progress(.95,'原始模型已保存')
    return {'scene_path':str(scene_path),'ply_path':str(ply),'backend':'original_3dgs'}
