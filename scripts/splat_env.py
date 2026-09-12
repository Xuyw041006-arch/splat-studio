"""Prepare/run offline CUDA tasks on Linux or Windows; configure a local GPU App.

System prerequisites are installed by the user (see the complete guide).
This helper installs Python packages and downloads pinned training resources.
It never starts training from the 'prepare' or 'local' commands.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from task_exchange import unpack_task

CUDA_IMPORTS='import torch,diff_gaussian_rasterization,simple_knn._C,fused_ssim; from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection,SamModel,SamProcessor; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name(0))'
def run(args,env=None,cwd=None):
    print('+', ' '.join(map(str,args)),flush=True)
    subprocess.run(list(map(str,args)),env=env,cwd=cwd,check=True)

def environment(base, colmap_root=None):
    env=dict(os.environ,PYTHONUTF8='1',PYTHONUNBUFFERED='1',MAX_JOBS='2',
             HF_HOME=str(base/'models'),SPLAT_GEOMETRY_HOME=str(base/'geometry'),
             SPLAT_3DGS_REPO=str(base/'compiler/vendor/gaussian-splatting'))
    nvcc=shutil.which('nvcc',path=env.get('PATH'))
    if not nvcc:raise ValueError('Missing nvcc. Install CUDA Toolkit 12.8 and add its bin to PATH.')
    version=subprocess.check_output([nvcc,'--version'],text=True)
    if 'release 12.8' not in version:raise ValueError('This environment recipe uses CUDA Toolkit 12.8; check nvcc --version.')
    env['CUDA_HOME']=str(Path(nvcc).resolve().parent.parent)
    if sys.platform=='win32':
        if colmap_root:
            folders=[colmap_root/'bin',colmap_root/'lib',colmap_root]
            env['PATH']=os.pathsep.join(str(f) for f in folders if f.is_dir())+os.pathsep+env.get('PATH','')
            plugins=colmap_root/'plugins'
            if plugins.is_dir():env['QT_PLUGIN_PATH']=str(plugins)
        colmap=shutil.which('colmap.exe',path=env['PATH'])
        if not colmap:raise ValueError('Need colmap.exe with its DLL paths, not only COLMAP.bat. Use --colmap-root.')
    else:
        env['QT_QPA_PLATFORM']='offscreen'
        colmap=shutil.which('colmap',path=env['PATH'])
        if not colmap:raise ValueError('Install COLMAP first (Ubuntu: apt install colmap).')
    run([colmap,'-h'],env)
    return env,colmap

def configure(runtime,base,geometry,colmap_root):
    if sys.platform not in {'linux','win32'}:raise ValueError('Full CUDA training needs Linux or Windows with NVIDIA. Use this helper on the GPU machine.')
    if not (3,11)<=sys.version_info[:2]<=(3,13):raise ValueError('Use standard Python 3.11–3.13, preferably 3.12 (not the app embedded Python).')
    for name in ['git','nvidia-smi','cl' if sys.platform=='win32' else 'g++']:
        if not shutil.which(name):raise ValueError('Missing '+name+'; see system prerequisites. On Windows use Developer PowerShell for VS 2022.')
    run(['nvidia-smi'])
    base.mkdir(parents=True,exist_ok=True)
    env,colmap=environment(base,colmap_root)
    # Preserve exact upstream source bytes on Windows even for older App tasks.
    # Scope this Git setting to setup subprocesses, never the user's global Git.
    git_index=int(env.get('GIT_CONFIG_COUNT','0'))
    env.update(GIT_CONFIG_COUNT=str(git_index+1))
    env['GIT_CONFIG_KEY_'+str(git_index)]='core.autocrlf'
    env['GIT_CONFIG_VALUE_'+str(git_index)]='false'
    venv=base/'env';py=venv/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    if not py.is_file():run([sys.executable,'-m','venv',venv])
    env['PATH']=str(py.parent)+os.pathsep+env['PATH']
    constraints=base/'constraints.txt'
    constraints.write_text('torch==2.11.0\ntorchvision==0.26.0\nnumpy==2.2.6\nsetuptools<82\n',encoding='utf-8')
    env['PIP_CONSTRAINT']=str(constraints)
    run([py,'-m','pip','install','--upgrade','pip','setuptools<82','wheel'],env)
    run([py,'-m','pip','install','torch==2.11.0','torchvision==0.26.0','--index-url','https://download.pytorch.org/whl/cu128'],env)
    run([py,'-m','pip','install','-r',runtime/'requirements-semantic.txt'],env)
    run([py,'-c','import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__,torch.cuda.get_device_name(0))'],env)
    compiler=base/'compiler/scripts';compiler.mkdir(parents=True,exist_ok=True)
    setup=runtime/'scripts/setup_upstream.py'
    sha=hashlib.sha256(setup.read_bytes()).hexdigest()
    stamp=base/'compiled.json';cache=json.loads(stamp.read_text()) if stamp.is_file() else {}
    repo=Path(env['SPLAT_3DGS_REPO'])
    arch=subprocess.check_output([str(py),'-c','import torch; print(torch.cuda.get_device_capability(0))'],env=env,text=True).strip()
    identity={'setup_sha256':sha,'python':str(py),'profile':'torch2.11-cu128','gpu_arch':arch}
    reuse=cache==identity and (repo/'train.py').is_file()
    if reuse:
        reuse=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()=='54c035f7834b564019656c3e3fcc3646292f727d'
        reuse=reuse and not subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True).strip()
        reuse=reuse and subprocess.run([str(py),'-c',CUDA_IMPORTS],env=env).returncode==0
    if not reuse:
        shutil.copyfile(setup,compiler/'setup_upstream.py')
        run([py,compiler/'setup_upstream.py','--cuda'],env)
        stamp.write_text(json.dumps(identity,indent=2),encoding='utf-8')
    if geometry:run([py,runtime/'scripts/setup_geometry.py','--root',base/'geometry'],env)
    if sys.platform=='linux':
        # Select CPU feature flags from the installed COLMAP version's help.
        folder=base/'bin';folder.mkdir(exist_ok=True)
        script=folder/'colmap'
        script.write_text('#!'+str(py)+'\nimport os,sys,subprocess\nREAL='+repr(colmap)+'''\nargs=sys.argv[1:]
choices={'feature_extractor':('FeatureExtraction.use_gpu','SiftExtraction.use_gpu'),'exhaustive_matcher':('FeatureMatching.use_gpu','SiftMatching.use_gpu')}
if args and args[0] in choices:
    choices=choices[args[0]]
    if not any(a.split('=')[0] in ['--'+o for o in choices] for a in args):
        p=subprocess.run([REAL,args[0],'-h'],capture_output=True,text=True)
        option=next((o for o in choices if o in p.stdout+p.stderr),None)
        if not option:raise SystemExit('COLMAP CPU option not found')
        args+=['--'+option,'0']
os.execv(REAL,[REAL]+args)
''',encoding='utf-8')
        script.chmod(0o755);env['PATH']=str(folder)+os.pathsep+env['PATH']
    run([py,'-c',CUDA_IMPORTS],env)
    # Do not persist unrelated environment variables or API credentials.
    keys=['PATH','CUDA_HOME','PYTHONUTF8','PYTHONUNBUFFERED','MAX_JOBS','HF_HOME','SPLAT_GEOMETRY_HOME','SPLAT_3DGS_REPO','QT_PLUGIN_PATH','QT_QPA_PLATFORM']
    state={'python':str(py),'env':{k:env[k] for k in keys if k in env},'platform':sys.platform}
    (base/'environment.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    return state

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ['prepare','local']:
        q=sub.add_parser(name);q.add_argument('--base',type=Path,required=True);q.add_argument('--colmap-root',type=Path)
        if name=='prepare':q.add_argument('--task',type=Path,required=True)
        else:q.add_argument('--source',type=Path,required=True);q.add_argument('--geometry',action='store_true')
    q=sub.add_parser('run');q.add_argument('--job',type=Path,required=True)
    q=sub.add_parser('launch');q.add_argument('--base',type=Path,required=True);q.add_argument('--app',type=Path,required=True)
    q=sub.add_parser('desktop-source');q.add_argument('--base',type=Path,required=True);q.add_argument('--source',type=Path,required=True)
    a=p.parse_args()
    if a.command=='prepare':
        base=a.base.resolve();batch=base/('task-'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]);batch.mkdir(parents=True)
        task=batch/'task.zip';shutil.copyfile(a.task.resolve(),task)
        extracted=batch/'unpacked';manifest=unpack_task(task,extracted)
        cfg=manifest['worker_config'];runtime=extracted/'runtime'
        count=len(json.loads((extracted/'project/project.json').read_text(encoding='utf-8'))['images'])
        if cfg.get('cloud_gpu')=='a100':
            names=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True)
            if 'A100' not in names:raise ValueError('App requested A100; use A100 or export a new task for the actual GPU.')
        geometry=cfg.get('priority_task_stage')!='review' and 2<=count<=12 and (cfg.get('sparse_completion','auto')!='none' or cfg.get('geometry_backend')=='dust3r')
        if geometry and not cfg.get('allow_geometry_download'):
            if not (base/'geometry').exists():raise ValueError('No geometry cache. Allow required resources in App and re-export.')
        state=configure(runtime,base,geometry,a.colmap_root.resolve() if a.colmap_root else None)
        env=dict(os.environ,**state['env']);run([state['python'],runtime/'scripts/cloud_worker.py',task,'--verify-only'],env)
        (batch/'job.json').write_text(json.dumps({'base':str(base),'runtime':str(runtime),'task':str(task),'work':str(batch/'run')},indent=2),encoding='utf-8')
        print('\nPrepared, not started. Job directory:\n'+str(batch)+'\nNext: python splat_env.py run --job "'+str(batch)+'"')
    elif a.command=='local':
        configure(a.source.resolve(),a.base.resolve(),a.geometry,a.colmap_root.resolve() if a.colmap_root else None)
        print('Local CUDA environment ready. Close the App, then use the launch command.')
    else:
        job=json.loads((a.job/'job.json').read_text(encoding='utf-8')) if a.command=='run' else None
        base=Path(job['base']) if job else a.base.resolve()
        state=json.loads((base/'environment.json').read_text(encoding='utf-8'))
        if state['platform']!=sys.platform:raise ValueError('This environment belongs to a different operating system.')
        env=dict(os.environ,**state['env'])
        if job:
            if Path(job['work']).exists():raise ValueError('Run directory exists. Download completed results or prepare a fresh task to retry; no automatic resume.')
            worker=Path(job['runtime'])/'scripts/cloud_worker.py'
            run([state['python'],worker,job['task'],'--verify-only'],env)
            run([state['python'],worker,job['task'],'--work-dir',job['work']],env)
            print('Outputs:',job['work'])
        elif a.command=='launch':
            if not a.app.resolve().is_file():raise ValueError('--app must point to Splat Studio.exe')
            env['SPLAT_PYTHON']=state['python'];run([a.app.resolve()],env)
        else:
            env['SPLAT_PYTHON']=state['python']
            run(['pnpm','run','desktop'],env,cwd=a.source.resolve())

if __name__=='__main__':main()
