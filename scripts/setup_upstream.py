"""Pinned upstream checkout. CUDA extensions install only when requested."""
from pathlib import Path
import json
import os
import subprocess
import sys
root=Path(__file__).resolve().parents[1];dest=root/'vendor/gaussian-splatting'
commit='54c035f7834b564019656c3e3fcc3646292f727d'
def run(*args,env=None,log_path=None,cwd=None):
    command=list(args)
    if log_path is None:
        return subprocess.run(command,check=True,env=env,cwd=cwd)
    log_path=Path(log_path);log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('w') as log:
        process=subprocess.Popen(command,env=env,cwd=cwd,stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
        code=process.wait()
    if code:
        raise RuntimeError(f'CUDA install command exited {code}; complete output: {log_path}')
    return subprocess.CompletedProcess(command,code)
if not (dest/'.git').exists() and not (dest/'train.py').exists():
    run('git','clone','https://github.com/graphdeco-inria/gaussian-splatting.git',str(dest))
if (dest/'.git').exists():
    run('git','-C',str(dest),'checkout',commit)
    run('git','-C',str(dest),'submodule','update','--init','--recursive')
else:
    import hashlib,json
    manifest=json.loads((root/'vendor/UPSTREAM.json').read_text())
    for name,digest in manifest['sha256'].items():
        if hashlib.sha256((dest/name).read_bytes()).hexdigest()!=digest:
            raise SystemExit('Bundled upstream source differs from pinned manifest: '+name)
if '--cuda' in sys.argv:
    import torch
    if not torch.cuda.is_available(): raise SystemExit('CUDA GPU required; original rasterizer has no MPS build.')
    logs=root/'work'/'cuda-setup-logs'
    run(sys.executable,'-m','pip','install','ninja','tqdm','plyfile','tensorboard','opencv-python','scipy',
        log_path=logs/'dependencies.log')
    # Build-only environment: preserve caller flags without changing the process
    # environment used later for training. Verified with CUDA 12.8 / torch 2.11.
    build_env=os.environ.copy()
    build_env['NVCC_PREPEND_FLAGS']=('-include cstdint '+build_env.get('NVCC_PREPEND_FLAGS','')).strip()
    build_env.setdefault('MAX_JOBS','2')
    major,minor=torch.cuda.get_device_capability()
    build_env.setdefault('TORCH_CUDA_ARCH_LIST',f'{major}.{minor}')
    build_info={'python':sys.version.split()[0],'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
                'nvcc_prepend_flags':build_env['NVCC_PREPEND_FLAGS'],'max_jobs':build_env['MAX_JOBS'],
                'cuda_arch_list':build_env['TORCH_CUDA_ARCH_LIST'],
                'compatibility':'Force-include standard cstdint for the pinned rasterizer header; no upstream source or algorithm changes.',
                'status':'building'}
    logs.mkdir(parents=True,exist_ok=True)
    (logs/'build-environment.json').write_text(json.dumps(build_info,indent=2))
    for sub in ['diff-gaussian-rasterization','simple-knn','fused-ssim']:
        package=dest/'submodules'/sub
        install=(sys.executable,'-m','pip','install','-v','--no-build-isolation',str(package))
        try:
            run(*install,env=build_env,log_path=logs/f'build-{sub}.log')
        except RuntimeError:
            # pip can hide the underlying nvcc error; a direct build exposes it.
            run(sys.executable,'setup.py','build_ext','--inplace',cwd=package,env=build_env,
                log_path=logs/f'direct-build-{sub}.log')
            run(*install,env=build_env,log_path=logs/f'install-after-build-{sub}.log')
    run(sys.executable,'-c','import torch,diff_gaussian_rasterization,simple_knn._C,fused_ssim; '
        'assert torch.cuda.is_available(); print("CUDA extensions imported:",torch.__version__)',
        log_path=logs/'import-check.log')
    build_info['status']='extensions_imported'
    (logs/'build-environment.json').write_text(json.dumps(build_info,indent=2))
