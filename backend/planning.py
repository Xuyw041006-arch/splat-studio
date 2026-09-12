"""Evidence-based capture triage. Estimates are ranges, never benchmark claims."""
from __future__ import annotations
import hashlib
import importlib.util
import math
import os
import shutil
from pathlib import Path
import cv2
import numpy as np

MODES = {
    'fast': {'iterations': 7000, 'resolution': 4, 'preview_steps': 48, 'preview_size': 64},
    'balanced': {'iterations': 15000, 'resolution': 2, 'preview_steps': 80, 'preview_size': 96},
    'fine': {'iterations': 22000, 'resolution': 1, 'preview_steps': 120, 'preview_size': 128},
}

def capabilities():
    import torch
    mps = bool(torch.backends.mps.is_available())
    cuda = bool(torch.cuda.is_available())
    extensions = all(importlib.util.find_spec(m) is not None for m in ('diff_gaussian_rasterization', 'simple_knn'))
    upstream = Path(os.environ.get('SPLAT_3DGS_REPO', Path(__file__).resolve().parents[1] / 'vendor/gaussian-splatting'))
    device = 'cuda' if cuda else 'mps' if mps else 'cpu'
    try: gpu_name=torch.cuda.get_device_name(0) if cuda else ''
    except Exception: gpu_name='NVIDIA GPU' if cuda else ''
    from .learned_geometry import geometry_availability
    geometry = geometry_availability()
    return {'device': device, 'gpu_name': gpu_name, 'mps': mps, 'cuda': cuda, 'torch': torch.__version__,
            'colmap': bool(shutil.which('colmap')), 'upstream_repository': (upstream/'train.py').is_file(),
            'original_3dgs': cuda and extensions and (upstream/'train.py').is_file(),
            'cuda_semantic_refinement': cuda and extensions and (upstream/'scene/gaussian_model.py').is_file(),
            'preview': True, 'joint_semantics': False, 'learned_geometry': geometry,
            'learned_completion': bool(os.environ.get('SPLAT_COMPLETION_COMMAND')),
            'vision_configured': bool(os.environ.get('SPLAT_VISION_MODEL'))}

def largest_component(nodes, edges):
    remaining=set(nodes); components=[]
    while remaining:
        reached={min(remaining)}
        while True:
            before=len(reached)
            for edge in edges:
                if edge['valid'] and (edge['i'] in reached or edge['j'] in reached):
                    reached.update((edge['i'],edge['j']))
            if len(reached)==before: break
        remaining-=reached;components.append(sorted(reached))
    return max(components,key=len) if components else []

def capture_pairs(count):
    """Retain the short-baseline links skipped by global uniform sampling.

    Every image is represented. Neighbor pairs keep capture sequences linked;
    global anchors add cross-sequence evidence without all-pairs matching.
    Connectivity remains a precheck, not calibrated camera coverage.
    """
    anchors=np.linspace(0,count-1,min(12,count),dtype=int).tolist()
    pairs={(i,j) for a,i in enumerate(anchors) for j in anchors[a+1:]}
    pairs.update((i,i+gap) for gap in (1,2) for i in range(count-gap))
    return sorted(pairs)

def retrieval_pairs(features, neighbors=8):
    """Find likely overlapping images, independent of upload order.

    Descriptor votes only propose pairs; the normal bidirectional match and
    RANSAC checks below must validate every edge before it affects triage.
    """
    valid=[(i,entry[1]) for i,entry in enumerate(features)
           if entry[1] is not None and len(entry[1])]
    if len(valid)<2:return []
    descriptors=np.concatenate([d for _,d in valid]).astype(np.float32)
    owners=np.concatenate([np.full(len(d),i,dtype=np.int32) for i,d in valid])
    index=cv2.flann_Index(descriptors,dict(algorithm=1,trees=4))
    pairs=set()
    for i,des in valid:
        queries=des[np.linspace(0,len(des)-1,min(96,len(des)),dtype=int)]
        nearest,_=index.knnSearch(queries,min(12,len(descriptors)),params={'checks':48})
        votes=np.zeros(len(features),dtype=np.int32)
        for row in owners[nearest]:votes[np.unique(row[row!=i])]+=1
        candidates=sorted((j for j in range(len(features)) if votes[j]),key=lambda j:(-votes[j],j))[:neighbors]
        pairs.update(tuple(sorted((i,j))) for j in candidates)
    index.release()
    return sorted(pairs)

def analyze_images(paths: list[str], config: dict | None=None):
    if len(paths) < 2:
        raise ValueError('至少需要两张有重叠区域的照片。单张图像需要另行配置单图生成模型。')
    config=config or {}
    all_hashes=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths]
    unique_count=len(set(all_hashes))
    samples = list(range(len(paths)))
    features, details, hashes = [], [], []
    sift = cv2.SIFT_create(nfeatures=1800)
    for idx in samples:
        p = paths[int(idx)]
        # imread's narrow Windows paths fail for Chinese account/folder names.
        gray = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_GRAYSCALE) if os.name == 'nt' else cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f'无法解码照片 {Path(p).name}')
        h, w = gray.shape
        gray = cv2.resize(gray, (round(w*min(1,800/max(w,h))), round(h*min(1,800/max(w,h)))))
        kp, des = sift.detectAndCompute(gray, None)
        features.append((kp, des, gray.shape))
        details.append({'name': Path(p).name, 'width': w, 'height': h,
                        'features': len(kp), 'sharpness': round(float(cv2.Laplacian(gray,cv2.CV_64F).var()),2)})
        hashes.append(all_hashes[int(idx)])
    edges=[]
    matcher=cv2.BFMatcher()
    pairs=set(capture_pairs(len(features)))
    if len(features)>12:pairs.update(retrieval_pairs(features))
    for i,j in sorted(pairs):
        ki,di,si=features[i]; kj,dj,sj=features[j]
        good=[]
        if di is not None and dj is not None and len(dj)>1:
            forward=[m for pair in matcher.knnMatch(di,dj,k=2) if len(pair)==2 for m,n in [pair] if m.distance<0.72*n.distance]
            reverse={m.queryIdx:m.trainIdx for pair in matcher.knnMatch(dj,di,k=2) if len(pair)==2 for m,n in [pair] if m.distance<0.72*n.distance}
            good=[m for m in forward if reverse.get(m.trainIdx)==m.queryIdx]
        inliers=0
        coverage=0.;homography_ratio=0.
        if len(good)>=8:
            a=np.float32([ki[m.queryIdx].pt for m in good]); b=np.float32([kj[m.trainIdx].pt for m in good])
            _,mask=cv2.findFundamentalMat(a,b,cv2.FM_RANSAC,1.5,0.99)
            inliers=int(mask.sum()) if mask is not None else 0
            if inliers:
                selected=np.asarray(mask).ravel().astype(bool)
                coverage=min(float(cv2.contourArea(cv2.convexHull(p[selected]))) / (s[0]*s[1]) for p,s in ((a,si),(b,sj)))
            _,hm=cv2.findHomography(a,b,cv2.RANSAC,2.)
            homography_ratio=float(hm.sum())/len(good) if hm is not None else 0.
        edges.append({'i':int(samples[i]),'j':int(samples[j]),'matches':len(good),'inliers':inliers,
                      'image_coverage':round(coverage,4),'homography_ratio':round(homography_ratio,4),
                      'planar_or_rotation_candidate':homography_ratio>.9,
                      'localized_support':coverage<.005,
                      'valid':inliers>=16 and hashes[i]!=hashes[j]})
    reached=largest_component([int(i) for i in samples],edges)
    warnings=[]
    if unique_count<len(paths): warnings.append('检测到完全重复照片；重复文件不增加有效视角。')
    if any(d['sharpness']<35 for d in details): warnings.append('部分照片清晰度偏低，建议补拍或剔除。')
    if len(reached)<len(samples): warnings.append('已检查的照片关联未覆盖全部图片；可检查拍摄顺序、重叠，或是否混入其他场景。')
    if len(paths)<=2: warnings.append('两个视角无法确定遮挡背面；补全依赖先验，不能用于准确测量。')
    if any(e['valid'] and e['planar_or_rotation_candidate'] for e in edges): warnings.append('部分匹配接近平面或纯旋转；重建阶段还需视差与深度检查，匹配多不代表深度可靠。')
    if any(e['valid'] and e['localized_support'] for e in edges): warnings.append('部分匹配仅覆盖很小的图像区域；可尝试验证局部几何，但不能据此认定整个场景已被观测。')
    return {'image_count':len(paths),'sample_count':len(samples),'checked_image_count':len(samples),
            'pair_selection':'neighbor_chain_retrieval_and_global_anchors','sampled_images':details,'pair_matches':edges,
            'input_sha256':all_hashes,'unique_image_count':unique_count,'duplicate_count':len(paths)-unique_count,
            'largest_component':reached,'intrinsics_sources':config.get('intrinsics_sources',[]),
            'connected_ratio':len(reached)/len(samples),'max_pair_inliers':max((e['inliers'] for e in edges if e['valid']),default=0),
            'duplicate_count_in_sample':len(hashes)-len(set(hashes)), 'warnings':warnings,
            'assessment':'仅为几何匹配预检；最终以相机注册率、三角化视差和重投影误差为准。'}

def make_plan(analysis: dict, config: dict, hardware: dict | None=None):
    if config.get('semantics'):
        from .semantic_profiles import resolve_semantic_profile
        config=resolve_semantic_profile(config)
    hardware=hardware or capabilities()
    mode=config.get('mode','balanced'); settings=MODES[mode]
    n=analysis['image_count']; span=config.get('view_span')
    unique=analysis.get('unique_image_count',n)
    extreme=unique<=2
    sparse=extreme or unique<12 or analysis['connected_ratio']<0.75 or (span is not None and span<90)
    strategy='two_view_completion' if extreme else 'sparse_regularized' if sparse else 'standard_3dgs'
    device=config.get('device','auto')
    device=hardware['device'] if device=='auto' else device
    backend='original_3dgs' if device=='cuda' and hardware['original_3dgs'] else 'torch_preview'
    multiplier={'fast':1.,'balanced':6.25,'fine':23.3}[mode]
    if backend=='original_3dgs':
        seconds=(settings['iterations']/30)*max(0.6,n/80)**0.25
    else:
        # Calibrated on a tiny synthetic 96-splat/64px fixture on this Mac.
        # Scaling N*H*W is appropriate only to this dense reference rasterizer.
        seconds=0.75*multiplier*({'mps':1,'cuda':0.7,'cpu':1.6}.get(device,2))
    overhead=max(3,n*1.2)+ (n*8 if config.get('semantics') else 0)
    warnings=list(analysis['warnings'])
    geometry_backend=config.get('geometry_backend','auto')
    completion=config.get('sparse_completion','auto')
    geometry=hardware.get('learned_geometry',{})
    download_allowed=config.get('allow_geometry_download') is True
    learned_ready=bool((geometry.get('available') and (not geometry.get('requires_download') or download_allowed)) or
        (download_allowed and geometry.get('source_available') and not geometry.get('missing_dependencies')))
    blockers=[]
    if not 2<=n<=12:blockers.append('learned_view_count')
    if unique!=n:blockers.append('learned_duplicate_views')
    supplied=config.get('intrinsics_original')
    has_supplied=supplied is not None and len(supplied)>0
    intrinsic_sources=config.get('intrinsics_sources',analysis.get('intrinsics_sources',[]))
    approximate=(isinstance(intrinsic_sources,list) and len(intrinsic_sources)==n and
                 all(source in {'heuristic','exif_35mm_approximate'} for source in intrinsic_sources))
    if (has_supplied and not approximate) or 'user_calibrated' in intrinsic_sources:
        blockers.append('learned_calibration_unsupported')
    learned_usable=learned_ready and not blockers
    classical=analysis['max_pair_inliers']>=16
    shared_initialization=(backend!='original_3dgs' or sparse or geometry_backend in {'sfm','dust3r'} or
        completion=='learned_visible' or 'user_calibrated' in intrinsic_sources or
        2<=n<=6 or (not hardware.get('colmap',False) and n<=80))
    classical_capacity=n<=80 if shared_initialization else bool(hardware.get('colmap',False))
    classical_attempt=classical and classical_capacity
    learned_requested=(geometry_backend=='dust3r' or completion=='learned_visible' or
        (completion=='auto' and (extreme or (n<=6 and analysis['connected_ratio']<.75) or
                                (not classical and geometry_backend=='auto'))))
    reason=[]
    if extreme: reason.append('仅有两个独立视角')
    elif n<12: reason.append('输入少于 12 个视角')
    if analysis['connected_ratio']<.75: reason.append('采样匹配图覆盖不足')
    if span is not None and span<90: reason.append('用户提供的拍摄覆盖角度不足 90°')
    if not reason: reason.append('照片数量与采样重叠满足常规初始化条件')
    can_reconstruct=unique>=2 and (classical_attempt or (geometry_backend=='auto' and completion!='none' and learned_usable))
    if geometry_backend=='dust3r': can_reconstruct=unique>=2 and learned_usable
    if backend=='original_3dgs' and shared_initialization and has_supplied:
        sizes=config.get('original_sizes',[])
        if len(sizes)==n and any(not np.allclose([k[0][2],k[1][2]],np.asarray(size)/2,atol=1e-6,rtol=0.)
                                for k,size in zip(supplied,sizes)):
            can_reconstruct=False
            warnings.append('固定原始 3DGS 投影目前要求主点位于照片中心；当前标定为偏心主点，不能忽略后训练。')
    if not classical_capacity:
        warnings.append('当前共享照片初始化最多支持 80 张图；更大场景需要标准 CUDA/COLMAP 路径，或先减少输入。')
    completion_status='not_requested' if completion=='none' else 'automatic_if_needed'
    if learned_requested:
        completion_status='ready' if learned_usable else 'unavailable'
        if blockers:
            if 'learned_view_count' in blockers:warnings.append('当前 DUSt3R 路径支持 2 至 12 张输入；此次输入数量不适用学习式初始化或补全。')
            if 'learned_duplicate_views' in blockers:warnings.append('DUSt3R 不接受重复照片；请去重后使用学习式路径，重复项可由传统初始化排除。')
            if 'learned_calibration_unsupported' in blockers:warnings.append('当前 DUSt3R 不能约束用户标定内参；将保留标定并使用传统几何，不会忽略标定或假称已执行学习式补全。')
        elif not learned_ready:
            warnings.append('学习式几何资源未就绪，自动补全暂不可执行；可用的传统几何仍会保留。先运行模型安装脚本，并允许首次下载或配置本地权重。')
        else:
            warnings.append('自动补全将使用 DUSt3R 的可见区域几何先验，并检查跨视角支持；推断点单独标记，未拍摄背面保持未知。')
    if unique<2: warnings.append('少于两张不同照片，无法用重复图像恢复真实视差。请补拍。')
    semantic_requested=config.get('semantic_refinement','projection')
    semantic_available=backend=='original_3dgs' and hardware.get('cuda_semantic_refinement',False)
    semantic_planned='cuda_iterative' if semantic_requested=='cuda_iterative' and semantic_available else 'projection'
    if config.get('semantics') and semantic_requested=='cuda_iterative':
        if not semantic_available:
            warnings.append('后置多轮语义优化需要 NVIDIA CUDA 和原始 3DGS 依赖；当前路径将回退原投影融合，不执行多轮优化。')
        else:
            warnings.append(f"重建后以 {config.get('semantic_steps',1200)} 步为语义基础预算，按有效视角覆盖调整；冻结完整 RGB 模型，运行后分阶段实测耗时。")
    if backend=='torch_preview': warnings.append('Mac/CPU 本地路径是低分辨率、小规模优化预览；精细成品请使用原始 CUDA/Colab 后端。')
    if extreme: warnings.append('默认输出照片支持的可见部分。启用补全时，会单独标记 inferred 区域；未配置学习式补全模型时不会声称完成遮挡背面。')
    if config.get('semantic_strategy') in ('joint','joint_experimental'):
        warnings.append('联合语义训练需要带一致 ID 的逐像素监督。本版提供加权重建与后置几何融合，不包含 SAGA/LaGa 联合训练复现。')
    return {'strategy':strategy,'sparse':sparse,'extreme':extreme,'reason':'；'.join(reason),
            'geometry_backend_requested':geometry_backend,'sparse_completion_requested':completion,
            'automatic_completion_status':completion_status,'learned_geometry_ready':learned_ready,
            'learned_geometry_input_eligible':not blockers,'learned_geometry_blockers':blockers,
            'can_attempt_classical':classical_attempt,'geometry_initialization_path':'shared_photo_initialization' if shared_initialization else 'colmap_photo_initialization',
            'completion_recommended':extreme,'recommended_backend':backend,
            'semantic_refinement_requested':semantic_requested,'planned_semantic_method':semantic_planned if config.get('semantics') else None,
            'cuda_semantic_refinement_available':semantic_available,
            'device':device,'settings':settings,'estimated_seconds':[round(seconds*0.5+overhead),round(seconds*2.5+overhead)],
            'estimate_basis':'准备与训练的粗略规划范围，非本任务实测或端到端承诺。'+('CUDA 按固定 30 步/秒启发式估算，未针对当前 GPU 标定。' if backend=='original_3dgs' else '根据此 Mac 的小型合成训练基准估算，真实照片耗时会变化。')+('语义附加成本是保守占位；已分析的掩码会缓存复用，此项可能重复高估。' if config.get('semantics') else '')+'不包含模型下载和未知学习式补全时间。动态 ETA 仅估计当前训练循环，不包含后续验证、保存和语义融合。',
            'can_reconstruct':can_reconstruct, 'warnings':warnings,
            'suggested_capture':'保持曝光稳定，补拍相邻且重叠的视角；多拍被遮挡面，避免仅原地旋转镜头。'}
