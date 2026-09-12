#!/usr/bin/env python3
"""Export an explicitly sampled SH0 app preview from a full original 3DGS PLY.

Standalone dependencies: numpy and plyfile. Pillow is not required. No application import,
CUDA, model weights, rendering, or implicit pre-sampling. Membership arrays must
refer to the complete source PLY in its unchanged original vertex order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData

SH_C0=0.28209479177387814
FIELDS=["x","y","z","f_dc_0","f_dc_1","f_dc_2","opacity","scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"]
ALIASES={"stuffed bear":["毛绒熊","玩具熊","熊","teddy bear"],"bear nose":["熊鼻子"],
    "apple":["苹果"],"coffee mug":["咖啡杯"],"coffee":["咖啡"],"plate":["盘子"],
    "sheep":["羊"],"hooves":["蹄子"],"paper napkin":["纸巾"],"tea in a glass":["玻璃杯里的茶"],
    "bag of cookies":["饼干袋"],"three cookies":["三块饼干"],"yellow pouf":["黄色坐垫"]}


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path,value,compact=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    with temporary.open("w",encoding="utf-8") as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=None if compact else 2,
            separators=(",",":") if compact else None)
    temporary.replace(path)


def selected_indices(count,limit):
    if not 1<=limit<=250000:
        raise ValueError("--max-gaussians must be 1..250000 to fit the app import limit")
    if count<1:
        raise ValueError("The source PLY is empty")
    # Same selection regardless of labels/priority. Sampling happens exactly once.
    return np.linspace(0,count-1,min(count,limit),dtype=np.int64)


def camera_bridge(filename):
    if filename is None:
        return [],{"status":"absent"}
    raw=json.loads(Path(filename).read_text())
    if not isinstance(raw,list):
        raise ValueError("Expected original upstream cameras.json list")
    converted=[]
    for i,record in enumerate(raw):
        rotation=np.asarray(record["rotation"],dtype=float)
        center=np.asarray(record["position"],dtype=float)
        if rotation.shape!=(3,3) or center.shape!=(3,) or not np.isfinite(rotation).all() or not np.isfinite(center).all():
            raise ValueError(f"Invalid upstream camera transform {i}")
        if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-4) or not np.isclose(np.linalg.det(rotation),1,atol=1e-4):
            raise ValueError(f"Camera {i} rotation is not a proper orthonormal C2W rotation")
        width,height=int(record["width"]),int(record["height"])
        fx,fy=float(record["fx"]),float(record["fy"])
        if min(width,height,fx,fy)<=0 or not all(math.isfinite(v) for v in (fx,fy)):
            raise ValueError(f"Invalid camera calibration {i}")
        c2w=np.eye(4);c2w[:3,:3]=rotation;c2w[:3,3]=center
        w2c=np.eye(4);w2c[:3,:3]=rotation.T;w2c[:3,3]=-rotation.T@center
        name=str(record.get("img_name",record.get("image_name",i)))
        converted.append({"image_index":i,"image_name":name,"image_path":name,
            "width":width,"height":height,"intrinsics":[[fx,0.,width/2],[0.,fy,height/2],[0.,0.,1.]],
            "world_to_camera":w2c.tolist(),"camera_to_world":c2w.tolist(),
            "source_camera_id":record.get("id",i),"projection":"upstream centered pinhole FOV convention"})
    return converted,{"status":"converted","count":len(converted),"source_sha256":sha256(filename),
        "convention":"Upstream cameras.json position/rotation are camera-to-world; R_w2c=R_c2w.T, t_w2c=-R_c2w.T@position.",
        "principal_point":"Assumed image center, as the upstream FOV renderer; original off-center COLMAP principal points are not recoverable from this cameras.json.",
        "image_paths":"Source image names only; photographs are not bundled in the preview."}


def load_memberships(npz_path,classes_path,source_count,indices):
    if npz_path is None and classes_path is None:
        return [],np.empty((len(indices),0),dtype=bool),[],{"status":"not_supplied"}
    if npz_path is None or classes_path is None:
        raise ValueError("Provide both semantic membership NPZ and semantic classes JSON")
    catalog=json.loads(Path(classes_path).read_text())
    classes=catalog["classes"]
    if not isinstance(classes,list) or not all(isinstance(v,str) for v in classes) or len(classes)!=len(set(classes)):
        raise ValueError("semantic_classes.json must contain distinct class strings")
    if catalog.get("gaussian_count")!=source_count:
        raise ValueError("Semantic source Gaussian count differs from FULL PLY; refusing an ambiguous mapping")
    preview=[];counts=[]
    with np.load(npz_path,allow_pickle=False) as archive:
        expected={f"class_{i}" for i in range(len(classes))}
        if set(archive.files)!=expected:
            raise ValueError("Membership NPZ keys must exactly match class_0..class_(N-1)")
        for i,label in enumerate(classes):
            values=archive[f"class_{i}"]
            if values.shape!=(source_count,):
                raise ValueError(f"Membership {label} has shape {values.shape}, expected ({source_count},)")
            if values.dtype.kind not in "biu" or not np.isin(values,[0,1]).all():
                raise ValueError(f"Membership {label} must be a binary boolean/integer array")
            # Index the ORIGINAL-length membership by the SAME once-only indices.
            preview.append(values[indices].astype(bool,copy=False));counts.append(int(np.count_nonzero(values)))
    matrix=np.stack(preview,axis=1) if classes else np.empty((len(indices),0),dtype=bool)
    return classes,matrix,counts,{"status":"class_membership_loaded","membership_sha256":sha256(npz_path),
        "classes_sha256":sha256(classes_path),"source_order_declaration":catalog.get("order","not recorded"),
        "alignment_validation":"Full PLY count, every membership length, class key order and identical selected_indices verified. Source PLY identity still depends on supplying the matching evaluation artifacts.",
        "confidence":"Not present in these arrays; no Gaussian/object confidence or per-view support is fabricated.",
        "granularity":"Category unions. Distinct same-category instances are merged; no instance-level SAGA result or part hierarchy is claimed."}


def export_preview(ply_path,output,limit=100000,npz_path=None,classes_path=None,cameras_path=None,priority_labels=(),copy_ply=False):
    ply_path=Path(ply_path).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    stamp=time.perf_counter()
    ply=PlyData.read(str(ply_path),mmap="r");vertices=ply["vertex"].data
    missing=set(FIELDS)-set(vertices.dtype.names or ())
    if missing:
        raise ValueError("Not original 3DGS PLY; missing "+", ".join(sorted(missing)))
    count=len(vertices);indices=selected_indices(count,limit)
    # Read direct structured fields; do NOT call app read_ply which pre-samples.
    raw=np.column_stack([vertices[name][indices] for name in FIELDS]).astype(np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("Selected Gaussian parameters contain NaN/Infinity; no indices were silently filtered")
    positions=raw[:,:3];colors=np.clip(.5+SH_C0*raw[:,3:6],0,1)
    log_opacity=raw[:,6];log_scales=raw[:,7:10];quaternions=raw[:,10:14]
    opacities=1/(1+np.exp(-np.clip(log_opacity,-30,30)))
    scales=np.exp(np.clip(log_scales,-20,10))
    norms=np.linalg.norm(quaternions,axis=1)
    if np.any(norms<1e-12):
        raise ValueError("Selected PLY has a zero rotation quaternion")
    rotations=quaternions/norms[:,None]
    classes,memberships,source_class_counts,semantic_meta=load_memberships(npz_path,classes_path,count,indices)
    priority={s.strip().casefold() for s in priority_labels if s.strip()}
    if priority-set(label.casefold() for label in classes):
        raise ValueError("Priority labels must exist in supplied semantic_classes.json")
    preview_class_counts=memberships.sum(axis=0).astype(int).tolist()
    ids=[f"class-{i:03d}" for i in range(len(classes))]
    # Do not populate the editable tree with unsupported, all-zero query classes.
    objects=[{"id":"scene","label":"场景 · 类别分组预览","level":"scene","parent_id":None,
        "children":[ids[i] for i,n in enumerate(source_class_counts) if n>0],"visible":True,
        "gaussian_count":len(indices),"priority":False,"source":"full_cuda_ply_preview"}]
    for i,label in enumerate(classes):
        if source_class_counts[i]==0:
            continue
        objects.append({"id":ids[i],"label":label,"aliases":ALIASES.get(label.casefold(),[]),"level":"object",
            "semantic_granularity":"category_union","parent_id":"scene","children":[],"visible":True,
            "gaussian_count":preview_class_counts[i],"source_gaussian_count":source_class_counts[i],
            "priority":label.casefold() in priority,"source":"saved_class_membership", "grounded":True})
    gaussians=[]
    for row,original_index in enumerate(indices):
        record={"position":positions[row].tolist(),"color":colors[row].tolist(),"scale":scales[row].tolist(),
            "rotation":rotations[row].tolist(),"opacity":float(opacities[row]),"source_index":int(original_index),
            "source":"unknown"}
        member_ids=[ids[i] for i in np.flatnonzero(memberships[row])]
        if member_ids:
            record["semantic_ids"]=member_ids
            # For overlapping class unions, no unjustified primary owner is chosen.
            if len(member_ids)==1:
                record["object_id"]=member_ids[0]
                record["semantic_path"]=["scene",member_ids[0]]
        gaussians.append(record)
    if cameras_path is None:
        candidate=ply_path.parent.parent.parent/"cameras.json"
        if candidate.is_file():
            cameras_path=candidate
    cameras,camera_meta=camera_bridge(cameras_path)
    full_ply=ply_path
    if copy_ply:
        full_ply=output/"full_model.ply"
        if full_ply.resolve()!=ply_path:
            shutil.copy2(ply_path,full_ply)
    ply_hash=sha256(ply_path)
    np.save(output/"selected_indices.npy",indices,allow_pickle=False)
    clipping={"opacity_logit_values_clipped":int(np.count_nonzero((log_opacity<-30)|(log_opacity>30))),
        "scale_log_values_clipped":int(np.count_nonzero((log_scales<-20)|(log_scales>10))),
        "bounds":"Same preview activation bounds as app gaussian_io: opacity logits[-30,30], log-scales[-20,10]. Original PLY is unchanged."}
    warnings=[f"交互预览：原始 {count:,} 个高斯，确定性采样为 {len(indices):,} 个，仅显示 SH0 颜色；本画面不代表完整 CUDA 模型的渲染画质。",
        "语义按类别合并同类实例，没有实例级 SAGA 或可靠部件层级；置信度及观测来源未知。" if classes else "未加载语义结果；高斯置信度及观测来源未知，未标注为真实观测或补全。"]
    dropped=[classes[i] for i,n in enumerate(source_class_counts) if n>0 and preview_class_counts[i]==0]
    if dropped:
        warnings.append("以下类别在完整模型中存在，但本次预览未采样到："+"、".join(dropped))
    metadata={"backend":"CUDA 3DGS · 采样 SH0 预览","preview_sh_degree":0,"source_gaussian_count":count,
        "preview_gaussian_count":len(indices),"preview_sampled":len(indices)<count,"source_ply_sha256":ply_hash,
        "source_ply_path":str(ply_path),"full_ply_path":str(full_ply),"full_ply_copied":copy_ply,
        "sampling":{"method":"np.linspace across unchanged full PLY vertex order; exactly one sampling pass",
            "selected_indices_file":"selected_indices.npy","selected_indices_sha256":sha256(output/"selected_indices.npy"),
            "embedded_mapping":"Each gaussians[i].source_index equals selected_indices[i]",
            "label_dependent_sampling":False,"priority_dependent_sampling":False},
        "semantics":semantic_meta,"class_catalog":[{"index":i,"id":ids[i],"label":label,
            "source_count":source_class_counts[i],"preview_count":preview_class_counts[i]} for i,label in enumerate(classes)],
        "unsupported_classes":[classes[i] for i,n in enumerate(source_class_counts) if n==0],
        "classes_lost_in_preview":dropped,"priority_labels":sorted(priority),"priority_selection_only":True,
        "priority_note":"Selection metadata only; exporting does not train, refine or modify any Gaussian.",
        "camera_bridge":camera_meta,"activation_clipping":clipping,"warnings":warnings,
        "quality_reference":"Use original full PLY with its trained higher-order SH and original CUDA renderer for high-quality evaluation; do not measure benchmark PSNR/IoU on this sampled SH0 preview.",
        "source_note":"PLY has no observed/inferred provenance or confidence; source='unknown', confidence fields deliberately absent.",
        "export_script_sha256":sha256(Path(__file__)),"exported_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    scene={"gaussians":gaussians,"objects":objects,"cameras":cameras,"metadata":metadata}
    save_json(output/"scene.json",scene,compact=True)
    summary={"scene_json":str(output/"scene.json"),"scene_json_bytes":(output/"scene.json").stat().st_size,
        "source_gaussian_count":count,"preview_gaussian_count":len(indices),"editable_class_count":len(objects)-1,
        "camera_count":len(cameras),"source_ply_sha256":ply_hash,"full_ply_path":str(full_ply),
        "full_ply_copied":copy_ply,"export_wall_seconds":time.perf_counter()-stamp,
        "warnings":warnings,"source_artifacts":{"ply":str(ply_path),"semantic_membership":str(npz_path) if npz_path else None,
            "semantic_classes":str(classes_path) if classes_path else None,"cameras":str(cameras_path) if cameras_path else None}}
    save_json(output/"export_manifest.json",summary)
    (output/"README.txt").write_text("在 Splat Studio 点击导入，选择 scene.json。可缩放、旋转、按类别查找/隐藏/删除并撤销。\n"
        "\n"+"\n".join(warnings)+"\n\n"
        "selected_indices.npy 和每个高斯 source_index 保存了预览与完整 PLY 的对应关系。\n"
        "类别包括多个同类实例时，隐藏或删除类别会一起作用于这些实例。\n"
        "重点标记只是选择状态，不代表导出过程执行了精细重建。\n"
        "完整高质量模型："+str(full_ply)+"\n"
        +( "full_model.ply 已原样复制；保留它用于完整高斯和高阶 SH 的查看。\n" if copy_ply else "原始 PLY 保持不变但未复制到本导出目录；从 Colab 下载时务必另外保存它，或使用 --copy-ply。\n"),encoding="utf-8")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--max-gaussians",type=int,default=100000)
    parser.add_argument("--semantics",type=Path,help="Evaluation directory containing semantic_membership.npz and semantic_classes.json")
    parser.add_argument("--semantic-membership",type=Path)
    parser.add_argument("--semantic-classes",type=Path)
    parser.add_argument("--cameras",type=Path,help="Original benchmark model cameras.json; auto-detected next to point_cloud when omitted")
    parser.add_argument("--priority-label",action="append",default=[])
    parser.add_argument("--copy-ply",action="store_true",help="Copy the full untouched source PLY into the export directory as full_model.ply")
    args=parser.parse_args()
    if args.semantics and (args.semantic_membership or args.semantic_classes):
        parser.error("Use either --semantics or the explicit pair of semantic paths")
    membership=args.semantics/"semantic_membership.npz" if args.semantics else args.semantic_membership
    classes=args.semantics/"semantic_classes.json" if args.semantics else args.semantic_classes
    summary=export_preview(args.ply,args.output,args.max_gaussians,membership,classes,args.cameras,args.priority_label,args.copy_ply)
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
