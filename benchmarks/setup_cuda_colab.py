#!/usr/bin/env python3
"""Set up the exact original 3DGS CUDA source used by run_cuda_benchmark.py.

Run inside a Colab GPU runtime. Keeps the runtime's existing torch/torchvision;
does not mount Drive, read credentials, or install a replacement CUDA toolkit.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

COMMIT="54c035f7834b564019656c3e3fcc3646292f727d"
URL="https://github.com/graphdeco-inria/gaussian-splatting.git"


def run(command,cwd=None,env=None,log_path=None):
    print("COMMAND:"," ".join(map(str,command)),flush=True)
    if log_path is None:
        subprocess.run(list(map(str,command)),cwd=cwd,env=env,check=True)
        return
    log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open("w") as log:
        process=subprocess.Popen(list(map(str,command)),cwd=cwd,env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            log.write(line);log.flush();print(line,end="",flush=True)
        code=process.wait()
    if code:
        raise RuntimeError(f"Command exited {code}. Complete compiler/setup output: {log_path}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream",type=Path,default=Path("gaussian-splatting"))
    parser.add_argument("--max-jobs",type=int,default=2)
    parser.add_argument("--log-dir",type=Path,default=None)
    args=parser.parse_args();root=args.upstream.resolve();stamp=time.perf_counter()
    logs=(args.log_dir or root.parent/"cuda-setup-logs").resolve()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU runtime before setup")
    run(["nvidia-smi"],log_path=logs/"gpu.log");run(["nvcc","--version"],log_path=logs/"nvcc.log")
    if not root.exists():
        run(["git","clone","--no-checkout",URL,root],log_path=logs/"clone.log")
        run(["git","checkout","--detach",COMMIT],cwd=root,log_path=logs/"checkout.log")
    else:
        current=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
        if current != COMMIT:
            raise RuntimeError(f"Existing repository is at {current}, expected {COMMIT}; choose a new directory")
    run(["git","diff","--exit-code"],cwd=root)
    # No SIBR desktop viewer needed in Colab; avoid its large unrelated checkout.
    run(["git","submodule","update","--init","--recursive", "submodules/simple-knn",
        "submodules/diff-gaussian-rasterization","submodules/fused-ssim"],cwd=root,log_path=logs/"submodules.log")
    run([sys.executable,"-m","pip","install","ninja","plyfile","tqdm","Pillow","numpy"],log_path=logs/"dependencies.log")
    capability=torch.cuda.get_device_capability(0)
    environment={**os.environ,"MAX_JOBS":str(args.max_jobs),
        "TORCH_CUDA_ARCH_LIST":f"{capability[0]}.{capability[1]}",
        # CUDA 12.8 no longer supplies these integer declarations transitively
        # to the pinned rasterizer_impl.h. Force only a standard header include;
        # no source, optimization, rendering, or numerical behavior is changed.
        "NVCC_PREPEND_FLAGS":"-include cstdint " + os.environ.get("NVCC_PREPEND_FLAGS","")}
    for package in ("simple-knn","diff-gaussian-rasterization","fused-ssim"):
        package_dir=root/"submodules"/package
        install=[sys.executable,"-m","pip","install","-v","--no-build-isolation",package_dir]
        try:
            run(install,env=environment,log_path=logs/f"build-{package}.log")
        except RuntimeError:
            # Some pip/build wrappers hide the actual nvcc diagnostics. This
            # direct build preserves them and can reuse successfully built files.
            run([sys.executable,"setup.py","build_ext","--inplace"],cwd=package_dir,
                env=environment,log_path=logs/f"direct-build-{package}.log")
            run(install,env=environment,log_path=logs/f"install-after-build-{package}.log")
    run([sys.executable,"-c","import torch, torchvision, simple_knn._C, diff_gaussian_rasterization, fused_ssim; print('CUDA extensions import OK:', torch.cuda.get_device_name(0))"],log_path=logs/"import-check.log")
    details={"commit":COMMIT,"setup_wall_seconds":time.perf_counter()-stamp,"gpu":torch.cuda.get_device_name(0),
        "torch":torch.__version__,"cuda":torch.version.cuda,"compute_capability":list(capability),
        "nvcc_prepend_flags":environment["NVCC_PREPEND_FLAGS"],"logs":str(logs),
        "compiler_compatibility":"Force-include standard cstdint for pinned rasterizer header under CUDA 12.8; upstream source and reconstruction algorithm are unchanged.",
        "note":"Setup, dependency compilation and dataset downloads are excluded from reported model training time."}
    (root.parent/"cuda-setup-result.json").write_text(json.dumps(details,indent=2))
    print(json.dumps(details,indent=2),flush=True)
    print("Next: upload download_data.py + run_cuda_benchmark.py, download bonsai, then run --modes fast first.",flush=True)


if __name__=="__main__":
    main()
