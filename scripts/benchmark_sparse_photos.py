"""Photo-only 2/3/6-view geometry checks; never load dataset COLMAP files."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.capture import photo_metadata, geometry_config
from backend.reconstruction import initialize_geometry
from backend.planning import analyze_images

SELECTIONS={2:[0,8],3:[0,8,16],6:[0,4,8,12,16,20]}

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--report-dir',type=Path,required=True)
    parser.add_argument('--stage',choices=['sfm','auto','dust3r'],default='sfm')
    parser.add_argument('--device',choices=['cpu','mps','cuda'],default='mps')
    parser.add_argument('--dataset',choices=['all','bonsai','teatime'],default='all')
    parser.add_argument('--views',type=int,nargs='+',default=[2,3,6],choices=[2,3,6])
    args=parser.parse_args()
    sources={'bonsai':args.data_root/'mipnerf360/bonsai/images_4',
             'teatime':args.data_root/'lerf-ovs/lerf_ovs/teatime/images'}
    args.report_dir.mkdir(parents=True,exist_ok=True)
    for dataset,folder in sources.items():
        if args.dataset not in ('all',dataset):continue
        images=sorted(p for p in folder.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
        for count in args.views:
            key=f'{dataset}-{count}views-{args.stage}-{args.device}'
            report_path=args.report_dir/(key+'.json')
            if report_path.exists():raise FileExistsError('Preserve completed or failed evidence: '+str(report_path))
            isolated=args.work_dir/f'{dataset}-{count}views';isolated.mkdir(parents=True,exist_ok=True)
            items=[];paths=[]
            for i,index in enumerate(SELECTIONS[count]):
                source=images[index];target=isolated/f'{i:04d}.jpg';raw=source.read_bytes()
                with Image.open(source) as image:
                    upright=ImageOps.exif_transpose(image).convert('RGB')
                    upright.save(target,quality=96)
                    item={'name':target.name,'original_name':source.name,'width':upright.width,'height':upright.height,
                          'capture':photo_metadata(image,upright,raw),'uploaded_sha256':digest(target)}
                items.append(item);paths.append(str(target.resolve()))
            cfg=geometry_config({'images':items},{'geometry_backend':args.stage,
                'sparse_completion':'none' if args.stage=='sfm' else 'auto',
                'device':args.device,'sparse':True,'geometry_max_points':20000,'allow_geometry_download':False})
            record={'name':key,'dataset':dataset,'selected_sorted_indices':SELECTIONS[count],'inputs':items,
                'config':cfg,'protocol':{'initialization_input':'only selected EXIF-upright RGB photos and approximate focal metadata',
                'author_poses_loaded':False,'author_point_cloud_loaded':False,'ground_truth_loaded':False,
                'scope':'geometry execution and consistency checks; no held-out rendering quality or metric-scale claim'},
                'code_sha256':{p.name:digest(p) for p in [ROOT/'backend'/n for n in
                    ('capture.py','reconstruction.py','sparse_geometry.py','learned_geometry.py')]}}
            started=time.perf_counter()
            def progress(value,message):print(key,round(float(value),3),message,flush=True)
            try:
                record['analysis']=analyze_images(paths,cfg)
                result=initialize_geometry(paths,cfg,progress)
                output=args.report_dir/(key+'.npz')
                np.savez_compressed(output,points=result['points'],colors=result['colors'],
                                    confidence=result['confidence'],sources=np.asarray(result['sources']))
                cameras=args.report_dir/(key+'-cameras.json')
                cameras.write_text(json.dumps({'cameras':result['cameras'],'observations':result['observations']},allow_nan=False))
                record.update(status='completed',metadata=result['metadata'],point_count=len(result['points']),
                    observed_points=result['sources'].count('observed'),inferred_points=result['sources'].count('inferred'),
                    finite_geometry=bool(np.isfinite(result['points']).all()),
                    artifact_sha256={output.name:digest(output),cameras.name:digest(cameras)})
            except Exception as error:
                record.update(status='failed',error=str(error),error_code=getattr(error,'code',None),
                              diagnostics=getattr(error,'diagnostics',{}),traceback=traceback.format_exc())
            record['wall_seconds']=time.perf_counter()-started
            report_path.write_text(json.dumps(record,indent=2,ensure_ascii=False,allow_nan=False))
            print(json.dumps({'name':key,'status':record['status'],'seconds':record['wall_seconds'],
                              'points':record.get('point_count'),'error':record.get('error')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
