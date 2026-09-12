#!/usr/bin/env python3
"""Fixed-geometry semantic refinement; all mask training precedes held-out GT reads.

Uses the original CUDA rasterizer for differentiable three-channel groups.
This is an explicit-label research extension, not SAGA/LaGa feature reproduction.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.run_benchmark import load_colmap, load_annotations, mask_metrics, mean_metric, write_json
from benchmarks.evaluate_cuda_semantics import camera_for_record, read_training_masks
from backend.semantic_targets import build_semantic_targets, normalize_label

# Declared from query semantics BEFORE training/GT evaluation. Mask containment
# alone also proposed impossible relations, which must not enter the loss/tree.
SEMANTIC_RELATION_PRIOR={('coffee mug','coffee'):'contents_of',
    ('stuffed bear','bear nose'):'part_of',('sheep','hooves'):'part_of'}


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def split_records(args):
    records,points,provenance=load_colmap(args.scene_dir,Path(args.upstream)/'utils/read_write_model.py')
    del points
    by_name={Path(r['name']).name:r for r in records}
    raw=json.loads(Path(args.split).read_text())
    split=raw.get('split',raw)
    train,test=({Path(n).name for n in split[k]} for k in ('train','test'))
    if train&test or (train|test)-set(by_name):raise ValueError('Invalid RGB split')
    masks=read_training_masks(args.training_masks,by_name,train)
    return by_name,split,train,test,masks,provenance


def summarize_targets(built):
    return {k:v for k,v in built.items() if k!='per_frame'} | {'frame_observations':{
        frame:{label:{'positive_pixels':int(item['mask'].sum()),'shape':list(item['mask'].shape),
                       'confidence':float(item['confidence']),'mask_ids':item.get('mask_ids',[])}
               for label,item in classes.items()} for frame,classes in built['per_frame'].items()}}


def prepare_extra(args):
    from backend.semantics import discover_objects
    from benchmarks.benchmark_cuda_discovery import REFERENCE_CONFIG, MODELS
    from huggingface_hub import snapshot_download
    by_name,split,train,test,masks,prov=split_records(args)
    existing={Path(m['image_path']).name for m in masks}
    pool=sorted(train-existing)
    if args.extra_views>len(pool):raise ValueError('Insufficient additional training views')
    indices=np.linspace(0,len(pool)-1,args.extra_views).round().astype(int) if args.extra_views else []
    selected=[pool[i] for i in indices]
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'selection.json',{'old_views':sorted(existing),'new_views':selected,
        'selection':'evenly spaced names from actual RGB train minus original eight',
        'split_sha256':sha(args.split),'ground_truth_used':False})
    config=dict(REFERENCE_CONFIG)
    config['strict_query_labels']=bool(args.strict_query_labels)
    for key,model,revision in MODELS:
        config[key]=snapshot_download(model,revision=revision,local_files_only=True)
    started=time.perf_counter()
    all_masks=[];alignment_diagnostics=[]
    for item in masks:
        item=dict(item);item['mask_path']=str(Path(item['mask_path']).resolve());all_masks.append(item)
    for i,name in enumerate(selected):
        result=discover_objects([by_name[name]['file']],config)
        if result.get('available') is False:raise RuntimeError(result.get('warnings'))
        if result.get('device')!='cuda':raise RuntimeError('CUDA discovery required')
        alignment_diagnostics.extend(result.get('label_alignment',[]))
        for j,item in enumerate(result.get('masks',[])):
            p=out/'new-masks'/f'{i:03d}-{j:03d}.png';p.parent.mkdir(exist_ok=True)
            Image.fromarray(np.asarray(item['mask'],np.uint8)*255).save(p)
            all_masks.append({k:v for k,v in item.items() if k!='mask'}|{'mask_path':str(p)})
        print(json.dumps({'stage':'extra_masks','view':i+1,'total':len(selected),'image':name,'masks':len(result.get('masks',[]))}),flush=True)
    write_json(out/'training_masks.json',{'masks':all_masks,'ground_truth_used':False,'old_cache':str(Path(args.training_masks).resolve()),
        'old_cache_sha256':sha(args.training_masks),'old_count':len(masks),'old_views':sorted(existing),'new_views':selected,
        'train_image_names':sorted(existing|set(selected)),'seconds':time.perf_counter()-started,
        'config':config,'models':MODELS,'split_sha256':sha(args.split),
        'label_alignment':alignment_diagnostics})
    print('EXTRA_MASKS_COMPLETE',out/'training_masks.json',flush=True)


def model_and_renderer(args):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    sys.path.insert(0,str(Path(args.upstream).resolve()))
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    model=GaussianModel(3);model.load_ply(args.ply)
    tensors={}
    for name in ('_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity'):
        value=getattr(model,name);value.requires_grad_(False)
        tensors[name]=value.detach().cpu().clone()
    black=torch.zeros(3,device='cuda')
    pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,debug=False,antialiasing=False)
    def renderer(camera,colors):
        return render(camera,model,pipe,black,override_color=colors.contiguous(),use_trained_exp=False)['render']
    return model,renderer,tensors


def geometry_unchanged(model,original):
    import torch
    return all(not getattr(model,k).requires_grad and torch.equal(getattr(model,k).detach().cpu(),v) for k,v in original.items())


def initial_probabilities(args,labels,n):
    import torch
    p=torch.full((n,len(labels)),.05,dtype=torch.float32,device='cuda')
    source=Path(args.initial_membership)
    meta=json.loads(source.with_name('semantic_classes.json').read_text())
    raw=np.load(source,allow_pickle=False)
    for i,label in enumerate(meta['classes']):
        target=normalize_label(label)
        if target in labels:
            values=raw[f'class_{i}']
            if values.shape!=(n,):raise ValueError('Initial membership PLY count mismatch')
            col=labels.index(target)
            p[:,col]=torch.maximum(p[:,col],torch.from_numpy(values.astype(np.float32)).cuda()*.9+.05)
    return p


def train(args):
    import torch
    from backend.semantic_refinement import SemanticRefinementConfig,refine_semantics
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=args.resume)
    by_name,split,train_names,test_names,masks,provenance=split_records(args)
    built=build_semantic_targets(masks,allowed_frames=train_names)
    write_json(out/'targets.json',summarize_targets(built))
    labels=built['class_labels']
    model,renderer,original=model_and_renderer(args)
    n=len(model.get_xyz)
    observations=[]
    for frame,items in sorted(built['per_frame'].items()):
        name=Path(frame).name
        if name not in train_names or name in test_names:raise ValueError('Semantic train frame leakage')
        camera,_=camera_for_record(by_name[name],args.train_size,torch)
        h,w=camera.image_height,camera.image_width
        targets=np.zeros((len(labels),h,w),np.float32)
        observed=np.zeros(len(labels),bool);confidence=np.zeros(len(labels),np.float32)
        for label,item in items.items():
            c=labels.index(label)
            targets[c]=cv2.resize(item['mask'].astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)
            observed[c]=bool(targets[c].any());confidence[c]=item['confidence']
        observations.append({'frame_id':name,'camera':camera,'targets':torch.from_numpy(targets),
            'observed':observed,'confidence':confidence})
    edges=[]
    for e in built['hierarchy']['edges']:
        parent=e.get('parent_label',e.get('parent'))
        child=e.get('child_label',e.get('child'))
        if parent in labels and child in labels and (parent,child) in SEMANTIC_RELATION_PRIOR:
            edges.append((labels.index(parent),labels.index(child)))
    initial=initial_probabilities(args,labels,n)
    checkpoints=tuple(sorted({0,*[int(x) for x in args.checkpoints.split(',')],args.steps}))
    config=SemanticRefinementConfig.preset(args.preset,steps=args.steps,checkpoint_steps=checkpoints,seed=args.seed)
    metadata={'classes':labels,'gaussian_count':n,'ply':str(Path(args.ply).resolve()),'ply_sha256':sha(args.ply),
        'initial_membership_sha256':sha(args.initial_membership),'split':split,'split_sha256':sha(args.split),
        'training_masks_sha256':sha(args.training_masks),'training_views':sorted({o['frame_id'] for o in observations}),
        'training_mask_count':len(masks),'train_size':args.train_size,'preset':args.preset,'steps':args.steps,'seed':args.seed,
        'hierarchy_edges':edges,'class_semantics':'multi-label category unions; not instance IDs','geometry_frozen':True,
        'relation_prior':[{'parent':p,'child':c,'relation':r} for (p,c),r in SEMANTIC_RELATION_PRIOR.items()],
        'relation_prior_source':'query-word semantic compatibility declared before training; observed containment still required',
        'opacity_frozen':True,'gt_read_for_training':False,'model':torch.cuda.get_device_name(),'provenance':provenance,
        'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'backend/semantic_refinement.py',ROOT/'backend/semantic_targets.py']}}
    if args.resume:
        old=json.loads((out/'experiment.json').read_text())
        if old!=metadata:raise ValueError('Refuse to resume with changed provenance/configuration')
    else:write_json(out/'experiment.json',metadata)
    def checkpoint(step,probs,stats):
        folder=out/f'step_{step:05d}';folder.mkdir(exist_ok=True)
        values=probs.numpy().astype(np.float16)
        # float16 checkpoint storage is recorded; optimizer remains float32.
        np.savez_compressed(folder/'probabilities.npz',probabilities=values)
        np.savez_compressed(folder/'semantic_membership.npz',**{f'class_{i}':values[:,i]>=.5 for i in range(len(labels))})
        write_json(folder/'semantic_classes.json',{'classes':labels,'gaussian_count':n,'order':'unaltered input PLY order'})
        write_json(folder/'checkpoint.json',{'step':step,'storage_dtype':'float16','stats':stats,'geometry_unchanged':geometry_unchanged(model,original)})
        print(json.dumps({'stage':'checkpoint','step':step,'file':str(folder),'geometry_unchanged':True}),flush=True)
    def state_callback(step,state):
        temp=out/'resume.tmp.pt';torch.save(state,temp);temp.replace(out/'resume.pt')
    def progress(step,row):
        with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps({'stage':'semantic_training',**row}),flush=True)
    resume_state=torch.load(out/'resume.pt',map_location='cpu',weights_only=False) if args.resume else None
    probs,stats=refine_semantics(initial,observations,renderer,edges,config,checkpoint_callback=checkpoint,
        state_callback=state_callback,progress_callback=progress,resume_state=resume_state)
    if not geometry_unchanged(model,original):raise RuntimeError('Frozen geometry was modified')
    write_json(out/'training_stats.json',stats)
    write_json(out/'complete.json',{'complete':True,'geometry_unchanged':True,'checkpoints':list(checkpoints)})
    print('TRAIN_COMPLETE',out,flush=True)


def evaluate(args):
    import torch
    exp=Path(args.experiment).resolve();metadata=json.loads((exp/'experiment.json').read_text())
    if sha(args.ply)!=metadata['ply_sha256']:raise ValueError('Evaluation PLY changed')
    if sha(args.split)!=metadata['split_sha256']:raise ValueError('Evaluation split changed')
    by_name,split,train_names,test_names,masks,provenance=split_records(args)
    annotations=load_annotations(args.annotations,args.scene_dir)
    if set(annotations)&train_names or set(annotations)-test_names:raise ValueError('GT views are not strictly held out')
    model,renderer,original=model_and_renderer(args)
    labels=metadata['classes'];index={normalize_label(l):i for i,l in enumerate(labels)}
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    results=[]
    for folder in sorted(exp.glob('step_*')):
        probs=np.load(folder/'probabilities.npz',allow_pickle=False)['probabilities'].astype(np.float32)
        if probs.shape!=(len(model.get_xyz),len(labels)):raise ValueError('Probability shape mismatch')
        raw_membership=probs>=.5
        raw_unknown=float((~raw_membership.any(axis=1)).mean())
        representations=getattr(args,'representations','soft,editable_binary').split(',')
        if set(representations)-{'soft','editable_binary','hierarchy_closed_binary'}:
            raise ValueError('Unsupported semantic evaluation representation')
        for representation in representations:
            tensor=torch.from_numpy(probs).cuda()
            rows=[];dest=output/folder.name/representation;dest.mkdir(parents=True)
            if representation=='hierarchy_closed_binary':
                from backend.semantic_worker import _hierarchy_closure
                tensor=torch.from_numpy(_hierarchy_closure(probs,metadata['hierarchy_edges'])).cuda()
            if representation!='soft':tensor=(tensor>=.5).float()
            effective_membership=(tensor>=.5).cpu().numpy()
            counts={'raw_unknown_fraction':raw_unknown,
                'effective_unknown_fraction':float((~effective_membership.any(axis=1)).mean()),
                'raw_multilabel_fraction':float((raw_membership.sum(axis=1)>1).mean()),
                'effective_multilabel_fraction':float((effective_membership.sum(axis=1)>1).mean()),
                'membership_definition':'threshold .5 on all complete-PLY Gaussians including unseen/occluded points; not a GT accuracy metric',
                'per_class_raw_count':{l:int(raw_membership[:,i].sum()) for i,l in enumerate(labels)},
                'per_class_effective_count':{l:int(effective_membership[:,i].sum()) for i,l in enumerate(labels)},
                'raw_hierarchy_violations':sum(int((raw_membership[:,c]&~raw_membership[:,p]).sum()) for p,c in metadata['hierarchy_edges']),
                'effective_hierarchy_violations':sum(int((effective_membership[:,c]&~effective_membership[:,p]).sum()) for p,c in metadata['hierarchy_edges'])}
            for vi,name in enumerate(sorted(annotations)):
                camera,_=camera_for_record(by_name[name],args.eval_size,torch)
                truths={}
                for item in annotations[name]:
                    label=normalize_label(item['label']);mask=np.asarray(Image.open(item['mask_path']).convert('L'))>0
                    mask=cv2.resize(mask.astype(np.uint8),(camera.image_width,camera.image_height),interpolation=cv2.INTER_NEAREST).astype(bool)
                    truths[label]=mask|truths.get(label,np.zeros_like(mask))
                for label,truth in truths.items():
                    if label in index:
                        colors=tensor[:,index[label],None].expand(-1,3)
                        with torch.no_grad():score=renderer(camera,colors)[0].cpu().numpy()
                    else:score=np.zeros(truth.shape,np.float32)
                    prediction=score>=.5
                    rows.append({'image':name,'label':label,**mask_metrics(prediction,truth)})
                    safe=''.join(c if c.isalnum() else '_' for c in label)
                    Image.fromarray(np.concatenate([truth,prediction],axis=1).astype(np.uint8)*255).save(dest/f'{vi:02d}-{safe}.png')
            bylabel={l:{'views':sum(r['label']==l for r in rows),'mean_iou':mean_metric([r for r in rows if r['label']==l],'iou'),
                'mean_boundary_iou':mean_metric([r for r in rows if r['label']==l],'boundary_iou')} for l in sorted({r['label'] for r in rows})}
            groups={('touching_image_border' if flag else 'inside_image'):{'pairs':sum(r['gt_touches_image_border']==flag for r in rows),
                'mean_iou':mean_metric([r for r in rows if r['gt_touches_image_border']==flag],'iou'),
                'mean_boundary_iou':mean_metric([r for r in rows if r['gt_touches_image_border']==flag],'boundary_iou')} for flag in (False,True)}
            result={'step':int(folder.name.split('_')[1]),'representation':representation,'class_mean_iou':mean_metric(list(bylabel.values()),'mean_iou'),
                'mean_iou':mean_metric(rows,'iou'),'mean_boundary_iou':mean_metric(rows,'boundary_iou'),
                'per_label':bylabel,'image_border_groups':groups,'per_object_view':rows,'prediction_threshold':.5,
                'class_count':len(bylabel),'pairs':len(rows),'views':len(annotations),'eval_long_edge':args.eval_size,
                'training_run':str(exp),'probabilities_sha256':sha(folder/'probabilities.npz'),'gt_used_only_after_training':True,
                'metric':'held-out 2D projected category masks, no 3D/instance GT claim'}
            result['membership_diagnostics']=counts
            result['evaluator_sha256']=sha(__file__)
            write_json(dest/'summary.json',result);results.append(result)
            print(json.dumps({k:result[k] for k in ('step','representation','class_mean_iou','mean_boundary_iou')}),flush=True)
        del tensor,probs;gc.collect();torch.cuda.empty_cache()
    write_json(output/'summary.json',{'experiment':metadata,'checkpoints':results})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare-extra','train','evaluate'])
    p.add_argument('--upstream',required=True);p.add_argument('--scene-dir',required=True)
    p.add_argument('--split',required=True);p.add_argument('--training-masks',required=True)
    p.add_argument('--output',required=True);p.add_argument('--ply');p.add_argument('--initial-membership')
    p.add_argument('--annotations');p.add_argument('--experiment');p.add_argument('--extra-views',type=int,default=16)
    p.add_argument('--train-size',type=int,default=256);p.add_argument('--eval-size',type=int,default=256)
    p.add_argument('--preset',choices=['simple','improved'],default='improved')
    p.add_argument('--steps',type=int,default=2400);p.add_argument('--checkpoints',default='400,1200,2400');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--resume',action='store_true',help='Continue our own exact optimizer checkpoint with unchanged inputs/config')
    p.add_argument('--strict-query-labels',action='store_true',help='R2: align new teacher detections to exact declared query token spans')
    p.add_argument('--representations',default='soft,editable_binary',help='Evaluation only: soft, editable_binary, hierarchy_closed_binary')
    args=p.parse_args()
    {'prepare-extra':prepare_extra,'train':train,'evaluate':evaluate}[args.action](args)
if __name__=='__main__':main()
