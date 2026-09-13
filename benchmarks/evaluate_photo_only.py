#!/usr/bin/env python3
"""Evaluate photo-only runs on held-out RGB and human masks.

Test poses are registered to the frozen training SfM map with SIFT/PnP.
Neither author cameras/points nor held-out pixels update the trained model.
Semantic scores come from the saved learned NPZ, not a replacement fusion model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from benchmarks.run_benchmark import load_colmap, rgb_metrics, mask_metrics, write_json
from benchmarks.evaluate_cuda_semantics import camera_for_record

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def category_columns(catalog, labels):
    metadata=catalog['class_metadata']
    if len(metadata)!=len(catalog['classes']):raise ValueError('Catalog class metadata mismatch')
    output={label:[] for label in labels}
    for i,row in enumerate(metadata):
        names=set(row.get('source_labels',row.get('original_labels',[])))
        names.add(row.get('display_label',''))
        for name in names:
            if name in output:output[name].append(i)
    return output

def validate_photo_split(project, protocol, kind, images_dir):
    """Check original image identities, including App-renamed/reencoded inputs."""
    rows=project['images'];stored=[r['name'] for r in rows]
    originals=[r['original_name'] for r in rows]
    if len(set(stored))!=len(stored) or len(set(originals))!=len(originals):
        raise ValueError('Duplicate training image identity')
    if set(stored)!=set(protocol['train_'+kind]) or set(originals)!=set(protocol['original_'+kind]):
        raise ValueError('Actual App inputs differ from the frozen split')
    if set(originals)&set(protocol['evaluation_names']):
        raise ValueError('A held-out original photo is present under a renamed training file')
    for row in rows:
        expected=protocol['source_hashes'][row['original_name']]
        if row['capture']['source_sha256']!=expected or file_hash(Path(images_dir)/row['original_name'])!=expected:
            raise ValueError('Training source identity changed: '+row['original_name'])
    return {'input_views':len(rows),'original_names_disjoint':True,'source_hashes_verified':True}

def aggregate_rows(rows):
    """Report macro class IoU across all annotated frames, retaining zero predictions."""
    groups={}
    for row in rows:
        for item in row['objects']:
            groups.setdefault(item['label'],[]).append(item)
    by_class={}
    for label,items in sorted(groups.items()):
        intersection=sum(x['intersection'] for x in items)
        union=sum(x['union'] for x in items)
        by_class[label]={'views':len(items),'iou':intersection/union if union else None,
                         'mean_view_iou':float(np.mean([x['iou'] for x in items])),
                         'mean_boundary_iou':float(np.mean([x['boundary_iou'] for x in items])),
                         'mean_roi_psnr_db':float(np.mean([x['rgb']['psnr_db'] for x in items])),
                         'mean_roi_ssim':float(np.mean([x['rgb']['ssim'] for x in items]))}
    def mean(section,key):
        values=[r[section][key] for r in rows if r[section].get(key) is not None]
        return float(np.mean(values)) if values else None
    return {'evaluated_views':len(rows),'overall_psnr_db':mean('rgb','psnr_db'),
            'overall_ssim':mean('rgb','ssim'),'priority_roi_psnr_db':mean('priority_rgb','psnr_db'),
            'priority_roi_ssim':mean('priority_rgb','ssim'),
            'miou':float(np.mean([x['iou'] for x in by_class.values()])) if by_class else None,
            'mean_boundary_iou':float(np.mean([x['mean_boundary_iou'] for x in by_class.values()])) if by_class else None,
            'classes':by_class,
            'averaging':'RGB: equal weight per registered test frame. mIoU: sum intersections/unions over annotated views per class, then equal class weight.'}

def feature_map(records,points):
    sift=cv2.SIFT_create(nfeatures=10000)
    descriptors=[];point_ids=[];used={}
    for ordinal,record in enumerate(records):
        gray=cv2.imread(record['file'],cv2.IMREAD_GRAYSCALE)
        keypoints,desc=sift.detectAndCompute(gray,None)
        valid=np.asarray([int(i) in points for i in record['track_ids']])
        if desc is None or not valid.any():continue
        coords=np.asarray(record['track_xy'])[valid];ids=np.asarray(record['track_ids'])[valid]
        distances,nearest=cKDTree(coords).query(np.asarray([k.pt for k in keypoints]),k=1)
        keep=np.flatnonzero(distances<2.)
        for index in keep:
            pid=int(ids[nearest[index]])
            if used.get(pid,0)>=4:continue
            used[pid]=used.get(pid,0)+1
            descriptors.append(desc[index]);point_ids.append(pid)
        if ordinal%20==0:print('REGISTRATION_MAP',ordinal+1,len(records),len(descriptors),flush=True)
    if len(descriptors)<30:raise ValueError('Insufficient feature descriptors bound to training-only 3D points')
    return sift,np.asarray(descriptors,np.float32),np.asarray(point_ids,np.int64)

def solve_test_pose(xyz,xy,k,width,height):
    """Estimate a pose and one focal scale; test RGB is used only for registration."""
    xyz=np.asarray(xyz,np.float64);xy=np.asarray(xy,np.float64)
    if len(xyz)<30:raise ValueError('Fewer than 30 distinct matched training points')
    candidates=[]
    cv2.setRNGSeed(7)
    for scale in (.75,1.,1.25):
        kk=np.asarray(k,np.float64).copy();kk[0,0]*=scale;kk[1,1]*=scale
        ok,rotation,translation,inliers=cv2.solvePnPRansac(xyz,xy,kk,None,
            iterationsCount=10000,reprojectionError=4.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
        if ok and inliers is not None:candidates.append((len(inliers),rotation,translation,inliers.ravel(),scale))
    if not candidates:raise ValueError('PnP did not register this test view')
    _,rotation,translation,inliers,scale=max(candidates,key=lambda x:x[0])
    parameters=np.r_[rotation.ravel(),translation.ravel(),np.log(scale)]
    def residual(v,ids):
        kk=np.asarray(k,np.float64).copy();kk[0,0]*=np.exp(v[6]);kk[1,1]*=np.exp(v[6])
        uv=cv2.projectPoints(xyz[ids],v[:3],v[3:6],kk,None)[0].reshape(-1,2)
        return (uv-xy[ids]).ravel()
    bounds=([-np.inf]*6+[np.log(.5)],[np.inf]*6+[np.log(2.)])
    for _ in range(2):
        fit=least_squares(residual,parameters,args=(inliers,),bounds=bounds,loss='soft_l1',f_scale=2.,max_nfev=150)
        parameters=fit.x
        errors=np.linalg.norm(residual(parameters,np.arange(len(xyz))).reshape(-1,2),axis=1)
        inliers=np.flatnonzero(errors<=4.)
    kk=np.asarray(k,np.float64).copy();kk[0,0]*=np.exp(parameters[6]);kk[1,1]*=np.exp(parameters[6])
    transform=np.eye(4);transform[:3,:3]=cv2.Rodrigues(parameters[:3])[0];transform[:3,3]=parameters[3:6]
    positive=((xyz[inliers]@transform[:3,:3].T+transform[:3,3])[:,2]>0).mean() if len(inliers) else 0
    coverage=float(cv2.contourArea(cv2.convexHull(xy[inliers].astype(np.float32))))/(width*height) if len(inliers)>=3 else 0
    evidence={'matched_distinct_points':len(xyz),'inliers':len(inliers),
              'median_reprojection_pixels':float(np.median(errors[inliers])) if len(inliers) else None,
              'p95_reprojection_pixels':float(np.percentile(errors[inliers],95)) if len(inliers) else None,
              'image_hull_coverage':coverage,'positive_depth_fraction':float(positive),
              'focal_scale':float(np.exp(parameters[6]))}
    if len(inliers)<30 or evidence['median_reprojection_pixels']>3 or positive<.95 or coverage<.03:
        raise ValueError('Test registration quality gate failed: '+json.dumps(evidence))
    return transform,kk,evidence

def register_tests(dataset,images_dir,test_names,train_names,upstream):
    records,points,_=load_colmap(dataset,upstream/'utils/read_write_model.py')
    if set(r['name'] for r in records)-set(train_names):raise ValueError('Training geometry includes a held-out or unknown camera')
    sift,desc,pids=feature_map(records,points)
    matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=4),dict(checks=128))
    matcher.add([desc]);matcher.train()
    normalized=[]
    for r in records:
        k=np.asarray(r['intrinsics']);k[0]/=r['width'];k[1]/=r['height'];normalized.append(k)
    median_k=np.median(normalized,axis=0)
    registered=[];failures=[]
    for name in test_names:
        path=images_dir/name;gray=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if gray is None:raise ValueError('Missing test image '+name)
        height,width=gray.shape
        kp,query=sift.detectAndCompute(gray,None)
        matches=[];seen=set()
        for row in matcher.knnMatch(query,k=min(8,len(desc))) if query is not None else []:
            if not row:continue
            first=row[0];pid=int(pids[first.trainIdx])
            second=next((m for m in row[1:] if int(pids[m.trainIdx])!=pid),None)
            if second is not None and first.distance<.75*second.distance:matches.append((first.distance,first.queryIdx,pid))
        xyz=[];xy=[]
        for _,q,pid in sorted(matches):
            if pid in seen:continue
            seen.add(pid);xyz.append(points[pid].xyz);xy.append(kp[q].pt)
        k=median_k.copy();k[0]*=width;k[1]*=height
        try:
            transform,k,evidence=solve_test_pose(xyz,xy,k,width,height)
            registered.append({'name':name,'file':str(path),'width':width,'height':height,
                'intrinsics':k.tolist(),'world_to_camera':transform.tolist(),'registration':evidence})
            print('TEST_REGISTERED',name,json.dumps(evidence),flush=True)
        except ValueError as exc:
            failures.append({'name':name,'reason':str(exc)});print('TEST_NOT_REGISTERED',name,str(exc),flush=True)
    return registered,failures,[r['name'] for r in records]

def evaluate(args):
    import torch
    assert torch.cuda.is_available()
    protocol=json.loads(Path(args.protocol).read_text())
    train=protocol['train_'+args.kind];test=protocol['evaluation_names']
    assert set(train).isdisjoint(test)
    run=Path(args.run).resolve();output=Path(args.output).resolve()
    project_path=run.parent.parent/'project.json'
    input_audit=validate_photo_split(json.loads(project_path.read_text()),protocol,args.kind,args.images)
    if output.exists():raise FileExistsError('Use a new evaluation output')
    output.mkdir(parents=True)
    ply=run/'model/point_cloud/iteration_22000/point_cloud.ply'
    catalog_path=run/'semantic_refinement/catalog.json'
    catalog=json.loads(catalog_path.read_text());ply_sha=file_hash(ply)
    assert catalog['status']=='completed' and catalog['source_ply_sha256']==ply_sha
    assert not set(catalog['training_frames'])&set(test)
    upstream=Path(args.upstream).resolve()
    registered,failures,registered_train=register_tests(run/'dataset',Path(args.images),test,train,upstream)
    write_json(output/'registration.json',{'views':registered,'failed_views':failures,
        'purpose':'PnP to a frozen training-only map. This evaluates rendering, not metric geometry accuracy.'})
    if not registered:raise RuntimeError('No independent test cameras registered')
    sys.path.insert(0,str(upstream))
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from types import SimpleNamespace
    model=GaussianModel(catalog['sh_degree']);model.load_ply(str(ply))
    count=len(model.get_xyz);assert count==catalog['gaussian_count']
    model.active_sh_degree=catalog['sh_degree']
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False,antialiasing=False)
    background=torch.zeros(3,device='cuda')
    annotations=json.loads(Path(args.annotations).read_text())['images']
    labels=sorted({r['label'] for rows in annotations.values() for r in rows})
    cols=category_columns(catalog,labels)
    probability_path=catalog_path.parent/catalog['artifacts']['probabilities']['path']
    assert file_hash(probability_path)==catalog['artifacts']['probabilities']['sha256']
    with np.load(probability_path,allow_pickle=False) as pack:
        assert str(pack['source_ply_sha256'].item())==ply_sha
        assert pack['classes'].tolist()==catalog['classes']
        raw=pack['probabilities']
    assert raw.shape==(count,len(catalog['classes']))
    # Category union is fixed before seeing test scores; include every corresponding track.
    category=np.zeros((count,len(labels)),np.float32)
    for j,label in enumerate(labels):
        if cols[label]:category[:,j]=raw[:,cols[label]].max(axis=1)
    del raw
    scores=torch.from_numpy(category).cuda();del category
    rows=[]
    for record in registered:
        name=record['name'];camera,target=camera_for_record(record,max(record['width'],record['height']),torch)
        with torch.no_grad():
            prediction=render(camera,model,pipe,background,use_trained_exp=False)['render'].clamp(0,1).permute(1,2,0).cpu().numpy()
            semantic=[]
            for start in range(0,len(labels),3):
                colors=scores[:,start:start+3]
                if colors.shape[1]<3:colors=torch.nn.functional.pad(colors,(0,3-colors.shape[1]))
                semantic.extend(render(camera,model,pipe,background,override_color=colors.contiguous(),use_trained_exp=False)['render'].cpu().numpy()[:min(3,len(labels)-start)])
        frame_dir=output/Path(name).stem;frame_dir.mkdir()
        Image.fromarray(np.round(prediction*255).astype('uint8')).save(frame_dir/'rgb.png')
        Image.fromarray(np.round(target*255).astype('uint8')).save(frame_dir/'reference.png')
        objects=[];priority=np.zeros(target.shape[:2],bool)
        for annotation in annotations[name]:
            label=annotation['label'];gt=np.asarray(Image.open(annotation['mask_path']).convert('L'))>127
            assert gt.shape==target.shape[:2]
            predicted=semantic[labels.index(label)]>=.5
            rgb=rgb_metrics(prediction,target,gt);metric=mask_metrics(predicted,gt)
            objects.append({'label':label,**metric,'rgb':rgb,'region_tracks':len(cols[label])})
            slug=label.replace(' ','-').replace('/','-')
            Image.fromarray(predicted.astype('uint8')*255).save(frame_dir/(slug+'-prediction.png'))
            Image.fromarray(gt.astype('uint8')*255).save(frame_dir/(slug+'-truth.png'))
            if label in protocol['priority_labels']:priority|=gt
        row={'name':name,'rgb':rgb_metrics(prediction,target),'priority_rgb':rgb_metrics(prediction,target,priority),'objects':objects}
        rows.append(row);print('EVALUATED',name,'PSNR',row['rgb']['psnr_db'],'priority',row['priority_rgb']['psnr_db'],flush=True)
    result={'status':'completed','kind':args.kind,'mode':'fine','iterations':22000,'input_views':len(train),
        'registered_training_views':len(registered_train),'registered_training_names':registered_train,'input_audit':input_audit,
        'expected_test_views':len(test),'registered_test_views':len(registered),'failed_test_views':failures,
        'source_ply_sha256':ply_sha,'gaussian_count':count,'sh_degree':catalog['sh_degree'],
        'semantic_catalog_sha256':file_hash(catalog_path),'protocol_sha256':file_hash(args.protocol),
        'semantic_field':'raw learned probabilities; category max, fixed 0.5 threshold; complete cloud opacity preserved',
        'semantic_steps':catalog['step'],'semantic_coverage':catalog.get('coverage'),
        'registration_scope':'Test poses only; no RGB, semantic, or geometry updates from held-out views',
        'summary':aggregate_rows(rows),'frames':rows}
    assert file_hash(ply)==ply_sha
    write_json(output/'metrics.json',result);print('FINE_METRICS',json.dumps(result['summary']),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','output','protocol','images','annotations','upstream'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--kind',choices=['full','sparse'],required=True)
    evaluate(parser.parse_args())

if __name__=='__main__':main()
