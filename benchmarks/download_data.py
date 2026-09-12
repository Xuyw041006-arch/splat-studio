#!/usr/bin/env python3
"""Download official public benchmark scenes without fetching unrelated scenes.

Downloads use the standard library only; --annotations-only requires Pillow.
HTTP Range responses are checked to prevent accidentally
downloading the complete 11.7 GiB Mip-NeRF 360 archive. Archives are never executed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

MIP_URL = "https://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
LERF_URL = "https://drive.usercontent.google.com/download?id=1QF1Po5p5DwTjFHu6tnTeYs_G0egMVmHt&export=download&confirm=t"


class RemoteFile(io.RawIOBase):
    """Minimal seekable HTTPS reader with strict, bounded range requests."""

    def __init__(self, url: str):
        self.url = url
        self.position = 0
        self.transferred = 0
        self.cache_start = -1
        self.cache_data = b""
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            self.size = int(response.headers["Content-Length"])
            self.etag = response.headers.get("ETag")

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence == 0:
            self.position = offset
        elif whence == 1:
            self.position += offset
        elif whence == 2:
            self.position = self.size + offset
        else:
            raise ValueError(whence)
        return self.position

    def read(self, count=-1):
        if count < 0:
            count = self.size - self.position
        count = min(count, self.size - self.position)
        if count <= 0:
            return b""
        if count > 256 * 1024 * 1024:
            raise RuntimeError("Refusing an unbounded remote archive read")
        desired_start = self.position
        cache_end = self.cache_start + len(self.cache_data)
        if self.cache_start <= desired_start and desired_start + count <= cache_end:
            offset = desired_start - self.cache_start
            self.position += count
            return self.cache_data[offset:offset + count]
        # ZIP readers issue separate tiny header/name reads. Prefetch contiguous
        # scene data to avoid hundreds of extra HTTP requests.
        start = desired_start
        fetch_count = min(max(count, 4 * 1024 * 1024), self.size - start)
        for attempt in range(4):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{start + fetch_count - 1}"})
                with urllib.request.urlopen(req, timeout=120) as response:
                    if response.status != 206:
                        raise RuntimeError("Server ignored HTTP Range; refusing full archive download")
                    value = response.read(fetch_count)
                    if len(value) != fetch_count:
                        raise RuntimeError(f"Expected {fetch_count} bytes, received {len(value)}")
                self.position += count
                self.transferred += fetch_count
                self.cache_start = start
                self.cache_data = value
                return value[:count]
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)


def safe_extract(archive, entry, destination):
    relative = PurePosixPath(entry.filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe archive member: {entry.filename}")
    target = destination.joinpath(*relative.parts)
    if entry.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.stat().st_size != entry.file_size:
        payload = archive.read(entry)
        target.write_bytes(payload)
    return {"path": entry.filename, "bytes": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}


def prepare_annotations(root, scene):
    """Rasterize authors' evaluation polygons without creating train labels."""
    from PIL import Image, ImageDraw

    labels = root / "lerf_ovs" / "label" / scene
    mask_dir = root / "lerf_ovs" / "evaluation_masks" / scene
    mask_dir.mkdir(parents=True, exist_ok=True)
    frames = {}
    for label_path in sorted(labels.glob("*.json")):
        record = json.loads(label_path.read_text())
        info = record["info"]
        shape = (int(info["width"]), int(info["height"]))
        masks = {}
        for obj in record["objects"]:
            label = obj["category"]
            mask = masks.setdefault(label, Image.new("L", shape, 0))
            polygon = [tuple(point) for point in obj["segmentation"]]
            ImageDraw.Draw(mask).polygon(polygon, fill=255)
        rows = []
        for label, mask in sorted(masks.items()):
            safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", label)
            mask_path = mask_dir / f"{label_path.stem}--{safe_name}.png"
            mask.save(mask_path)
            rows.append({"label": label, "mask_path": str(mask_path.resolve())})
        frames[info["name"]] = rows
    if not frames:
        raise RuntimeError(f"No annotation JSON found at {labels}")
    output = root / "lerf_ovs" / f"{scene}-evaluation-annotations.json"
    output.write_text(json.dumps({"dataset": "LERF-OVS", "scene": scene,
        "purpose": "held-out evaluation only; never use these masks as training inputs",
        "source": "LangSplat official lerf_ovs.zip; category union of annotated polygons",
        "images": frames}, indent=2))
    print(f"Exported {sum(map(len, frames.values()))} masks in {len(frames)} held-out views to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["mipnerf360", "lerf-ovs"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--resolution", default="images_4", choices=["images", "images_2", "images_4", "images_8"])
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--local-archive", type=Path)
    parser.add_argument("--annotations-only", action="store_true", help="Export downloaded LERF GT polygons as PNG masks (requires Pillow), without network access")
    args = parser.parse_args()
    scene = args.scene or ("bonsai" if args.dataset == "mipnerf360" else "teatime")
    if args.annotations_only:
        if args.dataset != "lerf-ovs":
            parser.error("Only lerf-ovs has segmentation annotations")
        prepare_annotations(args.output, scene)
        return
    url = MIP_URL if args.dataset == "mipnerf360" else LERF_URL
    source = args.local_archive or RemoteFile(url)
    with zipfile.ZipFile(source) as archive:
        if args.list_only:
            for entry in archive.infolist():
                print(entry.file_size, entry.filename)
            return
        selected = []
        for entry in archive.infolist():
            name = entry.filename
            parts = PurePosixPath(name).parts
            if args.dataset == "mipnerf360":
                keep = len(parts) > 2 and parts[0] == scene and (parts[1] == args.resolution or parts[1] == "sparse" or parts[-1].lower().startswith(("license", "readme")))
            else:
                keep = scene in parts and ("images" in parts or "sparse" in parts or "label" in parts)
            if keep and not entry.is_dir():
                selected.append(entry)
        if not selected:
            raise RuntimeError("No matching scene files in archive; inspect --list-only")
        records = []
        print(f"Selected {len(selected)} files, {sum(x.file_size for x in selected) / 1024 ** 2:.1f} MiB uncompressed", flush=True)
        for i, entry in enumerate(selected):
            records.append(safe_extract(archive, entry, args.output))
            if i % 20 == 0 or i == len(selected)-1:
                print(f"{i+1}/{len(selected)} {entry.filename}", flush=True)
        manifest = {"dataset": args.dataset, "scene": scene, "source_url": url,
                    "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "transferred_bytes": getattr(source, "transferred", None),
                    "source_etag": getattr(source, "etag", None), "files": records}
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"{scene}-download-manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
