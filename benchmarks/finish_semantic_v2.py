#!/usr/bin/env python3
"""Evaluate editing closure and collect all predeclared semantic experiments.

No new fit, parameter selection, or change to existing experiment directories.
The production smoke run is a separate end-to-end worker validation.
"""
from pathlib import Path
import hashlib,json,subprocess,sys,time,traceback,zipfile
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/content');OUT=BASE/'semantic-v2-finalization';OUT.mkdir(exist_ok=True)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def status(phase,**kw):
    row={'phase':phase,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw}
    (OUT/'status.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
def command(args,log):
    with (OUT/log).open('a') as f:
        f.write('\nCOMMAND '+json.dumps([str(x) for x in args])+'\n');f.flush()
        subprocess.run([str(x) for x in args],stdout=f,stderr=subprocess.STDOUT,check=True)
try:
    for folder in ('semantic-v2-results','semantic-v2b-results'):
        if json.loads((BASE/folder/'status.json').read_text())['phase']!='complete':raise RuntimeError('Training/evaluation queue must finish first')
    cases=[]
    for folder in ('semantic-v2-results','semantic-v2b-results'):
        for exp in sorted((BASE/folder).glob('*/experiment.json')):cases.append(exp.parent)
    for exp in cases:
        meta=json.loads((exp/'experiment.json').read_text())
        dest=OUT/'closure-evaluation'/exp.name
        if (dest/'summary.json').exists():continue
        status('evaluating_hierarchy_closed_editing',case=exp.name)
        masks=BASE/('semantic-v2b-results' if exp.parent.name=='semantic-v2b-results' else 'semantic-v2-results')/'masks24/training_masks.json' if '24' in exp.name else BASE/'splat-benchmark-tools/training_masks.json'
        command([sys.executable,'-u',ROOT/'benchmarks/run_semantic_refinement.py','evaluate',
            '--upstream',BASE/'splat-benchmark/gaussian-splatting','--scene-dir',BASE/'data/lerf-ovs/lerf_ovs/teatime',
            '--split',BASE/'benchmark-results/teatime/split.json','--training-masks',masks,
            '--ply',meta['ply'],'--annotations',BASE/'splat-benchmark-tools/annotations.json',
            '--experiment',exp,'--output',dest,'--representations','hierarchy_closed_binary'],exp.name+'-closure.log')
    status('validating_production_cuda_worker')
    validation=BASE/'semantic-worker-cuda-validation'
    if not (validation/'complete.json').exists():
        command([sys.executable,'-u',ROOT/'benchmarks/validate_semantic_worker_cuda.py'],'production-worker.log')
    status('validating_fresh_initialization_default_budget')
    fresh=BASE/'semantic-worker-fresh-validation'
    if not (fresh/'complete.json').exists():
        command([sys.executable,'-u',ROOT/'benchmarks/validate_semantic_worker_cuda.py','--fresh-default',
                 '--output',fresh],'production-worker-fresh.log')
    status('collecting_all_results')
    include=[]
    for folder in ('semantic-v2-results','semantic-v2b-results','semantic-v2-finalization','semantic-worker-cuda-validation','semantic-worker-fresh-validation'):
        for p in sorted((BASE/folder).rglob('*')):
            if not p.is_file() or p.suffix=='.pt' or p.name.endswith('.tmp'):continue
            # Final full probabilities and every binary checkpoint are retained.
            # Intermediate float probabilities remain on Colab and are hashed in inventory.
            if p.name=='probabilities.npz' and p.parent.name.startswith('step_') and p.parent.name!='step_02400':continue
            include.append(p)
    for name in ('Splat-Studio-semantic-v2-tools.zip','Splat-Studio-semantic-v2b-tools.zip','Splat-Studio-semantic-validation-tools.zip'):
        p=BASE/name
        if p.exists():include.append(p)
    inventory=[]
    for folder in ('semantic-v2-results','semantic-v2b-results','semantic-worker-cuda-validation','semantic-worker-fresh-validation'):
        for p in sorted((BASE/folder).rglob('*.npz')):
            inventory.append({'path':str(p.relative_to(BASE)),'size_bytes':p.stat().st_size,'sha256':sha(p),'included':p in include})
    (OUT/'all-checkpoint-inventory.json').write_text(json.dumps(inventory,indent=2))
    include.append(OUT/'all-checkpoint-inventory.json')
    archive=BASE/'Splat-Studio-semantic-v2-results.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        records=[]
        for p in sorted(set(include)):
            rel=p.relative_to(BASE).as_posix();z.write(p,rel)
            records.append({'path':rel,'size_bytes':p.stat().st_size,'sha256':sha(p)})
        z.writestr('DELIVERY_MANIFEST.json',json.dumps({'included_files':records,
            'geometry':'Original full PLY unchanged; models remain in previous A100 results/runtime.',
            'selection':'All cases and every checkpoint metric; full final probabilities plus all binary checkpoints; no GT checkpoint selection.',
            'intermediate_float_checkpoints':'Remain in Colab and listed in all-checkpoint-inventory.json; optimizer resume.pt excluded for size.'},indent=2))
    digest=sha(archive)
    (OUT/'archive.json').write_text(json.dumps({'path':str(archive),'bytes':archive.stat().st_size,'sha256':digest,'files':len(records)},indent=2))
    status('complete',archive=str(archive),bytes=archive.stat().st_size,sha256=digest,files=len(records))
except Exception as exc:
    status('failed',error=str(exc),traceback=traceback.format_exc());raise
