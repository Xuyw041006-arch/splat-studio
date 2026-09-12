"""Portable-runtime smoke test. No model download, no training, no user projects."""
from pathlib import Path
import io
import json
import os
import sys
import tempfile
import traceback

def main():
    base = Path(__file__).resolve().parent
    runtime = base / 'splat-backend' / '_internal' / 'cloud-runtime'
    sys.path.insert(0, str(runtime))
    report = {'platform': sys.platform, 'python': sys.version, 'checks': [], 'gpu_training_tested': False}
    log_dir = Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'Splat-Studio-Diagnostics'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'self-test.json'
    with tempfile.TemporaryDirectory(prefix='splat-test-') as temp:
        os.environ.update(SPLAT_DATA_DIR=str(Path(temp)/'中文照片'), SPLAT_SESSION_TOKEN='self-test-only',
                          SPLAT_FRONTEND_DIR=str(base.parent/'frontend'),
                          HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          SPLAT_GEOMETRY_REPO=str(base.parent/'geometry'/'dust3r'),
                          MPLCONFIGDIR=str(Path(temp)/'matplotlib'))
        try:
            import torch, torchvision, cv2, numpy as np
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor
            from PIL import Image
            from fastapi.testclient import TestClient
            from backend.app import app
            from backend.cloud_tasks import verify_package
            report['torch'] = torch.__version__
            report['checks'].append('CPU + image + semantic imports')
            # Exercise a compiled torchvision operator, not just its Python module.
            assert torchvision.ops.nms(torch.tensor([[0.,0.,1.,1.]]),torch.tensor([1.]),.5).tolist() == [0]
            cv2.SIFT_create()
            report['checks'].append('native torch/torchvision/OpenCV operators')
            from backend.learned_geometry import _runtime
            _runtime(base.parent/'geometry'/'dust3r')
            import dust3r.cloud_opt
            report['checks'].append('pinned geometry imports (no weights or inference)')
            client = TestClient(app, headers={'x-splat-token':'self-test-only'})
            response = client.get('/api/health'); assert response.status_code == 200, response.text
            assert client.get('/').status_code == 200
            files=[]
            rng=np.random.default_rng(17)
            pixels=rng.integers(0,256,(160,240,3),dtype=np.uint8)
            for i in range(2):
                buf=io.BytesIO(); Image.fromarray(np.roll(pixels,i*8,axis=1)).save(buf,format='JPEG')
                files.append(('files',(f'view-{i}.jpg',buf.getvalue(),'image/jpeg')))
            response=client.post('/api/projects',files=files)
            assert response.status_code==200,response.text
            pid=response.json()['id']
            cfg={'execution_target':'cloud','cloud_provider':'server','cloud_gpu':'other','semantics':False}
            response=client.post(f'/api/projects/{pid}/analyze',json=cfg)
            assert response.status_code==200,response.text
            report['checks'].append('API + frontend + photo import + Unicode-path preflight')
            response=client.post(f'/api/projects/{pid}/cloud-jobs',json=cfg)
            assert response.status_code==200,response.text
            job=response.json(); downloaded=client.get(job['package_url'])
            assert downloaded.status_code==200
            task=Path(temp)/'task.zip';task.write_bytes(downloaded.content)
            verify_package(task,runtime)
            report['checks'].append('cloud task export + source/hash verification')
            report['status']='passed'
        except BaseException:
            report['status']='failed';report['error']=traceback.format_exc()
        log_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('Report:',log_path)
    return 0 if report['status']=='passed' else 1

if __name__=='__main__':
    raise SystemExit(main())
