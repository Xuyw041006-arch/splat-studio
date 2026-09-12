"""Build private Colab support package; no dataset RGB or model weights."""
from pathlib import Path
import json
import shutil
import zipfile
import hashlib

app = Path(__file__).resolve().parents[1]
workspace = app.parents[1]
staging = workspace / "work/cuda-semantic-tools/splat-benchmark-tools"
staging.mkdir(parents=True, exist_ok=True)
files = ["benchmarks/__init__.py", "benchmarks/evaluate_cuda_semantics.py", "benchmarks/run_benchmark.py",
         "backend/__init__.py", "backend/reconstruction.py", "backend/gaussian_io.py", "backend/semantics.py"]
for rel in files:
    dst = staging / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(app / rel, dst)
raw = json.loads((app / "benchmarks/results/teatime/training_masks.json").read_text())
train = []
for i, mask in enumerate(raw["masks"]):
    rel = Path("train-masks") / f"{i:03d}.png"
    (staging / rel).parent.mkdir(exist_ok=True)
    shutil.copy2(mask["mask_path"], staging / rel)
    train.append({k:v for k,v in mask.items() if k not in {"mask_path", "image_path"}} |
                 {"image_name":Path(mask["image_path"]).name, "mask_path":str(rel)})
(staging / "training_masks.json").write_text(json.dumps({"masks":train, "source":raw["model"], "prediction_seconds":raw["seconds"],
    "train_image_names":sorted({m["image_name"] for m in train}), "ground_truth_used":False}, indent=2))
gt_path = workspace / "work/datasets/lerf-ovs/lerf_ovs/teatime-evaluation-annotations.json"
gt = json.loads(gt_path.read_text())
gt_images = {}
count = 0
for image, records in gt["images"].items():
    gt_images[image] = []
    for record in records:
        rel = Path("gt-masks") / f"{count:03d}.png"
        (staging / rel).parent.mkdir(exist_ok=True)
        shutil.copy2(record["mask_path"], staging / rel)
        gt_images[image].append({**record, "mask_path":str(rel)})
        count += 1
(staging / "annotations.json").write_text(json.dumps({"images":gt_images,
    "source":"Official LangSplat LERF-OVS teatime evaluation polygons rasterized at original image size", "held_out_only":True}, indent=2))
split_src = workspace / "work/cuda-benchmark-split-dry-run/teatime/split.json"
if split_src.exists():
    shutil.copy2(split_src, staging / "expected-teatime-split.json")
manifest = {"purpose":"Private user-authorized Colab evaluation support; exclude dataset material from app release",
    "contents":{"code_files":files, "predicted_train_masks":len(train), "predicted_train_mask_images":sorted({m["image_name"] for m in train}),
                "heldout_gt_masks":count, "heldout_gt_images":sorted(gt_images)},
    "sources":{"train_masks":"Local cached GroundingDINO Tiny + SAM ViT Base predictions on 8 downloaded public LERF-OVS training images, no heldout inference",
               "ground_truth":"Official LangSplat LERF-OVS archive; downloaded evaluation polygons rasterized by benchmarks/download_data.py",
               "destination":"User-authorized Google Colab runtime; no model weights, personal images, credentials, or original RGB photographs in this package"},
    "core_sha256":{rel:hashlib.sha256((staging/rel).read_bytes()).hexdigest() for rel in files}}
(staging / "PACKAGE_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
(staging / "README.txt").write_text("""Private Colab benchmark support.
Run:
python /content/splat-benchmark-tools/benchmarks/evaluate_cuda_semantics.py --upstream /content/gaussian-splatting --ply /content/MODEL/point_cloud/iteration_22000/point_cloud.ply --scene-dir /content/DATA/teatime --split /content/OUTPUT/teatime/split.json --training-masks /content/splat-benchmark-tools/training_masks.json --annotations /content/splat-benchmark-tools/annotations.json --output /content/cuda-semantic-evaluation --eval-size 256
All PLY Gaussians are loaded, without sampling. Actual training split must exclude the six GT views. CPU memory use of semantic fusion scales with the full point count.
""")
archive = workspace / "work/cuda-semantic-tools.zip"
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
    for file in sorted(staging.rglob("*")):
        if file.is_file():
            zipped.write(file, file.relative_to(staging.parent))
print(json.dumps({"archive":str(archive), "bytes":archive.stat().st_size, "train_masks":len(train), "gt_masks":count}, indent=2))
