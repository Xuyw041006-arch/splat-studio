#!/usr/bin/env python3
"""Run the predeclared A100 semantic experiments independently of browser polling."""
from pathlib import Path
import json, subprocess, sys, time, traceback
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/content')
OUT=BASE/'semantic-v2-results'
OUT.mkdir(exist_ok=True)
SCRIPT=ROOT/'benchmarks/run_semantic_refinement.py'
UPSTREAM=BASE/'splat-benchmark/gaussian-splatting'
DATA=BASE/'data/lerf-ovs/lerf_ovs/teatime'
SPLIT=BASE/'benchmark-results/teatime/split.json'
MASKS=BASE/'splat-benchmark-tools/training_masks.json'
GT=BASE/'splat-benchmark-tools/annotations.json'
COMMON=['--upstream',str(UPSTREAM),'--scene-dir',str(DATA),'--split',str(SPLIT)]
def status(phase,**details):
    row={'phase':phase,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**details}
    p=OUT/'status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2));tmp.replace(p)
    print(json.dumps(row),flush=True)
def command(args,log):
    with (OUT/log).open('a') as f:
        f.write('\nCOMMAND '+json.dumps([str(a) for a in args])+'\n');f.flush()
        subprocess.run([str(a) for a in args],stdout=f,stderr=subprocess.STDOUT,check=True)
def geometry(mode):
    steps=15000 if mode=='balanced' else 22000
    return BASE/f'benchmark-results/teatime/{mode}/point_cloud/iteration_{steps}/point_cloud.ply'
def initializer(mode):return BASE/f'semantic-results/{mode}/semantic_membership.npz'
try:
    masks24=OUT/'masks24/training_masks.json'
    if not masks24.exists():
        status('prepare_16_additional_training_views')
        command([sys.executable,'-u',SCRIPT,'prepare-extra',*COMMON,'--training-masks',MASKS,'--extra-views','16','--output',OUT/'masks24'],'prepare-extra.log')
    cases=[('simple-8','balanced','simple',MASKS),('improved-8','balanced','improved',MASKS),
           ('improved-24','balanced','improved',masks24),('improved-24-fine','fine','improved',masks24)]
    # Freeze all planned cases before any new GT evaluation is started.
    plan={'cases':[{'name':n,'geometry':g,'preset':p,'masks':str(m)} for n,g,p,m in cases],
          'steps':2400,'checkpoints':[0,400,1200,2400],'semantic_train_long_edge':256,'seed':0,
          'gt_read_after_all_training':True,'protocol':str(ROOT/'docs/SEMANTIC_V2_PROTOCOL.zh-CN.md')}
    manifest=OUT/'run-plan.json'
    if manifest.exists() and json.loads(manifest.read_text())!=plan:raise ValueError('Existing run plan differs')
    manifest.write_text(json.dumps(plan,indent=2))
    for name,mode,preset,masks in cases:
        dest=OUT/name
        if (dest/'complete.json').exists():continue
        status('training_semantic_field',case=name)
        args=[sys.executable,'-u',SCRIPT,'train',*COMMON,'--training-masks',masks,'--ply',geometry(mode),
              '--initial-membership',initializer(mode),'--preset',preset,'--output',dest]
        if dest.exists():
            if not (dest/'resume.pt').exists():raise RuntimeError(f'Incomplete setup without resume state: {dest}')
            args+=['--resume']
        command(args,name+'.log')
    textfix=OUT/'textfix-only'
    if not (textfix/'summary.json').exists():
        status('evaluating_text_normalization_only')
        command([sys.executable,'-u',ROOT/'benchmarks/evaluate_cuda_semantics.py',*COMMON,'--ply',geometry('balanced'),
            '--training-masks',MASKS,'--annotations',GT,'--output',textfix,'--normalize-labels'],'textfix-only.log')
    for name,mode,preset,masks in cases:
        dest=OUT/'evaluation'/name
        if (dest/'summary.json').exists():continue
        status('evaluating_all_semantic_checkpoints',case=name)
        command([sys.executable,'-u',SCRIPT,'evaluate',*COMMON,'--training-masks',masks,'--ply',geometry(mode),
                 '--annotations',GT,'--experiment',OUT/name,'--output',dest],'evaluate-'+name+'.log')
    status('complete',training_cases=len(cases),checkpoints_per_case=4)
except Exception as exc:
    status('failed',error=str(exc),traceback=traceback.format_exc());raise
