#!/usr/bin/env python3
"""CPU-only comparison: direct256 rendering vs native rendering resized to256.

Consumes finished evaluate_cuda_full_resolution.py outputs. Reads original PNGs
and checkpoints' result metadata without editing them. Before each comparison,
resizes the native GT with the upstream PIL.resize default and requires decoded
pixel equality with the original256 GT. This controls both pose and target.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version


def load_benchmark(path):
    spec=importlib.util.spec_from_file_location("scale_analysis_benchmark",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resize_like_upstream(path,size):
    """Match pinned utils.general_utils.PILtoTorch: RGB Image.resize(size)."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize(size)).copy()


def compare_view(native,direct,metrics):
    with Image.open(direct["ground_truth"]) as opened:
        target=np.asarray(opened.convert("RGB")).copy()
        size=opened.size
    if size[0]!=256:
        raise ValueError(f"Expected original256 evaluation PNG, got {size}")
    native_gt=resize_like_upstream(native["ground_truth"],size)
    difference=np.abs(native_gt.astype(np.int16)-target.astype(np.int16))
    validation={"decoded_gt_pixels_equal":bool(np.array_equal(native_gt,target)),
                "max_gt_byte_difference":int(difference.max(initial=0)),
                "differing_gt_pixel_count":int(np.any(difference!=0,axis=2).sum()),
                "original_gt_rgb_sha256":hashlib.sha256(target.tobytes()).hexdigest(),
                "native_resized_gt_rgb_sha256":hashlib.sha256(native_gt.tobytes()).hexdigest()}
    if not validation["decoded_gt_pixels_equal"]:
        raise ValueError("Native-resized GT differs from original256 GT: "+json.dumps(validation))
    with Image.open(direct["render"]) as opened:
        direct_rgb=np.asarray(opened.convert("RGB")).copy()
    downsampled=resize_like_upstream(native["render"],size)
    if direct_rgb.shape!=target.shape:
        raise ValueError("Original direct render differs in shape from its GT")
    direct_metrics=metrics(direct_rgb/255.,target/255.)
    downsampled_metrics=metrics(downsampled/255.,target/255.)
    return {"image_name":direct["image_name"],"width":size[0],"height":size[1],
            "direct_256":direct_metrics,"native_downsampled_256":downsampled_metrics,
            "delta_psnr_db":downsampled_metrics["psnr_db"]-direct_metrics["psnr_db"],
            "delta_ssim":downsampled_metrics["ssim"]-direct_metrics["ssim"],"validation":validation},(target,direct_rgb,downsampled)


def describe(row,threshold):
    delta=row["delta_psnr_db"]
    if delta>=threshold:
        return "Native rendering followed by identical resizing is better for this checkpoint; this supports an effect from the resolution used during rendering, but does not isolate the fixed covariance floor as its sole cause."
    if delta<=-threshold:
        return "Direct256 rendering is better for this checkpoint; native rendering followed by resizing does not explain this model as suffering a256 rendering penalty."
    return "This comparison is below the declared descriptive gain threshold; do not infer a large rendering-scale penalty from this result."


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-resolution-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--benchmark-runner",type=Path,default=Path(__file__).with_name("run_cuda_benchmark.py"))
    parser.add_argument("--material-psnr-gain",type=float,default=.5)
    args=parser.parse_args()
    source,output=args.full_resolution_root.resolve(),args.output.resolve()
    if output==source or source in output.parents or output in source.parents:
        raise ValueError("Analysis output must be separate from the original full-resolution outputs")
    if args.material_psnr_gain<=0:
        raise ValueError("Gain threshold must be positive")
    benchmark=load_benchmark(args.benchmark_runner.resolve())
    cases=[source/scene/mode/"result.json" for scene in ["bonsai","teatime"] for mode in ["fast","balanced","fine"]]
    cases.append(source/"teatime/balanced-priority/result.json")
    if any(not path.exists() for path in cases):
        raise FileNotFoundError("Wait until all seven full-resolution evaluations complete")
    output.mkdir(parents=True,exist_ok=True)
    results=[]
    target_hashes={}
    for filename in cases:
        record=json.loads(filename.read_text())
        source_model=Path(record["source_model"]).resolve()
        if output==source_model or source_model in output.parents or output in source_model.parents:
            raise ValueError("Analysis must not write inside an original model result tree")
        native={r["image_name"]:r for r in record["evaluation"]["per_view"]}
        direct={r["image_name"]:r for r in record["previous_256_evaluation"]["per_view"]}
        if len(native)!=len(record["evaluation"]["per_view"]) or len(direct)!=len(record["previous_256_evaluation"]["per_view"]):
            raise ValueError("Duplicate view names cannot be matched safely")
        if set(native)!=set(direct):
            raise ValueError("Native and original256 view names differ")
        destination=output/record["scene"]/record["mode"]
        destination.mkdir(parents=True,exist_ok=True)
        names=sorted(direct)
        previews=set(np.linspace(0,len(names)-1,min(3,len(names))).round().astype(int).tolist())
        rows=[]
        for index,name in enumerate(names):
            try:
                row,images=compare_view(native[name],direct[name],benchmark.rgb_metrics)
            except ValueError as exc:
                benchmark.write_json(destination/"validation_failure.json",{"image_name":name,"error":str(exc),"native":native[name],"direct":direct[name]})
                raise
            hash_key=(record["scene"],name)
            actual_hash=row["validation"]["original_gt_rgb_sha256"]
            if target_hashes.setdefault(hash_key,actual_hash)!=actual_hash:
                raise ValueError("Original256 GT differs across modes")
            rows.append(row)
            if index in previews:
                width,height=images[0].shape[1],images[0].shape[0]
                canvas=Image.new("RGB",(width*3,height+24),"white")
                draw=ImageDraw.Draw(canvas)
                for col,(image,label) in enumerate(zip(images,["Same256 GT","Direct render@256","Native render -> resize256"])):
                    canvas.paste(Image.fromarray(image),(col*width,24))
                    draw.text((col*width+4,5),label,fill="black")
                canvas.save(destination/f"{index:03d}-comparison.png")
        mean=lambda key,metric:float(np.mean([r[key][metric] for r in rows]))
        row={"scene":record["scene"],"mode":record["mode"],"checkpoint_sha256":record["source_checkpoint_sha256"],
             "native_size":record["downloaded_base_size"],"test_views":len(rows),
             "direct_256":{"mean_psnr_db":mean("direct_256","psnr_db"),"mean_ssim":mean("direct_256","ssim")},
             "native_downsampled_256":{"mean_psnr_db":mean("native_downsampled_256","psnr_db"),"mean_ssim":mean("native_downsampled_256","ssim")},
             "delta_psnr_db":float(np.mean([r["delta_psnr_db"] for r in rows])),"delta_ssim":float(np.mean([r["delta_ssim"] for r in rows])),
             "views_with_positive_psnr_delta":sum(r["delta_psnr_db"]>0 for r in rows),
             "all_native_resized_gt_equal_original_256":True,"per_view":rows}
        row["interpretation"]=describe(row,args.material_psnr_gain)
        benchmark.write_json(destination/"result.json",row)
        results.append(row)
        print(json.dumps({k:row[k] for k in ["scene","mode","delta_psnr_db","delta_ssim","views_with_positive_psnr_delta","test_views"]}),flush=True)
    summary={"experiment":"CPU postprocessing only: same checkpoint, same held-out camera and identical256 GT; direct low-resolution rendering versus native render followed by resizing",
             "resizing":"PIL RGB Image.resize((target_width,target_height)), default resampler, identical to pinned upstream PILtoTorch",
             "pillow_version":pillow_version,"descriptive_psnr_gain_threshold_db":args.material_psnr_gain,
             "threshold_note":"A reporting heuristic, not a statistical significance test. Every raw per-view delta is retained.",
             "no_additional_gpu_work":True,"original_files_modified":False,"results":results,
             "limitation":"A positive gain identifies a difference between render resolutions for the same trained representation. It does not alone isolate the0.3px covariance floor, exclude all other renderer effects, or validate changes to training."}
    benchmark.write_json(output/"summary.json",summary)
    with (output/"summary.csv").open("w",newline="") as stream:
        fields=["scene","mode","test_views","direct_256_psnr","native_downsampled_256_psnr","delta_psnr_db","direct_256_ssim","native_downsampled_256_ssim","delta_ssim","views_improved"]
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({"scene":row["scene"],"mode":row["mode"],"test_views":row["test_views"],
                "direct_256_psnr":row["direct_256"]["mean_psnr_db"],"native_downsampled_256_psnr":row["native_downsampled_256"]["mean_psnr_db"],
                "delta_psnr_db":row["delta_psnr_db"],"direct_256_ssim":row["direct_256"]["mean_ssim"],
                "native_downsampled_256_ssim":row["native_downsampled_256"]["mean_ssim"],"delta_ssim":row["delta_ssim"],
                "views_improved":row["views_with_positive_psnr_delta"]})
    lines=["# 渲染分辨率对照", "", "同一 checkpoint、相同相机、逐像素相同的256宽GT；全部为已有PNG的CPU后处理，没有再次训练或渲染。", "",
           "| 场景 | 模式 | 直接256 PSNR | 原尺寸渲染再缩256 PSNR | 差值 | 改善视角 |", "|---|---|---:|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['scene']} | {row['mode']} | {row['direct_256']['mean_psnr_db']:.3f} | {row['native_downsampled_256']['mean_psnr_db']:.3f} | {row['delta_psnr_db']:+.3f} | {row['views_with_positive_psnr_delta']}/{row['test_views']} |")
    lines += ["", f"描述性判据：平均PSNR差值达到{args.material_psnr_gain:g}dB视为明显变化，此阈值不是统计显著性检验。", "",
              "若原尺寸渲染后缩图明显更好，则同一个模型在直接低分辨率渲染时存在质量损失；该对照能定位渲染分辨率相关影响，但不能单独证明固定协方差下限是唯一原因。若没有明显改善，则不应凭机制推导声称找到根因。完整原始差值与SSIM保存在JSON/CSV。"]
    (output/"REPORT.zh-CN.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    main()
