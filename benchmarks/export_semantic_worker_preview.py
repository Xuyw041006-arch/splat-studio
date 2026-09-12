#!/usr/bin/env python3
"""Export indexed preview from a verified production semantic worker sidecar.

Preview sampling and SH0 rendering are explicit; no full model is overwritten.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.gaussian_io import read_ply,write_scene
from backend.semantic_sidecar import bind_semantic_sidecar
from backend.semantic_worker import _sha256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',default='/content/semantic-worker-cuda-validation/worker')
    p.add_argument('--ply',default='/content/benchmark-results/teatime/balanced/point_cloud/iteration_15000/point_cloud.ply')
    p.add_argument('--output',default='/content/semantic-v2-previews')
    args=p.parse_args();out=Path(args.output);out.mkdir(exist_ok=False)
    folder=Path(args.worker);catalog=json.loads((folder/'catalog.json').read_text())
    labels=catalog['classes'];sha=_sha256(args.ply)
    if catalog['source_ply_sha256']!=sha:raise ValueError('Worker / PLY mismatch')
    with np.load(folder/'probabilities.npz',allow_pickle=False) as data:
        if str(data['source_ply_sha256'])!=sha or data['classes'].tolist()!=labels:raise ValueError('Sidecar identity mismatch')
        probs=data['probabilities']
    edges=[{'parent_idx':labels.index(e['parent']),'child_idx':labels.index(e['child']),
        'relation':e['relation'],'source':e['source']} for e in catalog['hierarchy_edges']]
    for limit in (5000,100000):
        scene=read_ply(args.ply,max_points=limit)
        scene['metadata'].update({'backend':'original_3dgs','semantic_preview':'independent production-worker CUDA smoke, 400 steps',
            'preview_warning':'Sampled browser SH0 preview. Full alpha-composited CUDA evaluation is separate; this is not the full PLY.',
            'source_ply_sha256':sha})
        bound=bind_semantic_sidecar(scene,catalog,probs>=.5,membership_classes=labels,probabilities=probs,approved_edges=edges)
        name=f'teatime-production-semantic-400-preview-{limit}.json'
        write_scene(bound['scene'],out/name)
        (out/f'diagnostics-{limit}.json').write_text(json.dumps(bound['diagnostics'],indent=2))
        print(json.dumps({'path':str(out/name),'points':len(bound['scene']['gaussians']),'bytes':(out/name).stat().st_size}),flush=True)
if __name__=='__main__':main()
