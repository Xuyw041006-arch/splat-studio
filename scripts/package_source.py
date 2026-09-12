#!/usr/bin/env python3
"""Build a small source ZIP for the Splat Studio notebooks.

Create an auditable current-source archive:
  python scripts/package_source.py --output /path/to/Splat-Studio-source.zip

The ZIP contains source, notebooks, docs, and setup scripts for pinned CUDA dependencies. It excludes datasets, model weights, results, builds, and
Git metadata. Symlinks are never followed. No dependency installation occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path


EXCLUDED_DIRECTORIES = frozenset({
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "release", "release-v2", "build", "dist",
    "work", "vendor", "data", "datasets", "checkpoints", "models", "weights",
    "results", "outputs", "output", "SIBR_viewers", ".DS_Store",
})
EXCLUDED_SUFFIXES = frozenset({
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe", ".dmg", ".msi",
    ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".bin", ".ply",
    ".npz", ".npy", ".zip", ".gz", ".xz", ".bz2", ".7z", ".tar",
})
EXCLUDED_RELATIVE_PATHS = (Path("vendor/gaussian-splatting/assets"),)


def excluded(path: Path) -> bool:
    return (
        any(path == prefix or prefix in path.parents for prefix in EXCLUDED_RELATIVE_PATHS)
        or
        any(part in EXCLUDED_DIRECTORIES or part.endswith((".app", ".egg-info")) for part in path.parts)
        or path.suffix.lower() in EXCLUDED_SUFFIXES
        or path.name in {".DS_Store", ".env", "SOURCE_MANIFEST.json"}
        or (path.name.startswith(".env.") and path.name not in {".env.example", ".env.sample", ".env.template"})
    )


def source_files(root: Path):
    for folder, directories, filenames in os.walk(root, followlinks=False):
        folder = Path(folder)
        directories[:] = sorted(name for name in directories
            if not name.startswith("release") and not (folder / name).is_symlink()
            and not excluded((folder / name).relative_to(root)))
        for name in sorted(filenames):
            path = folder / name
            if not path.is_symlink() and not excluded(path.relative_to(root)):
                yield path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-file-mib", type=float, default=32)
    parser.add_argument("--max-total-mib", type=float, default=256)
    parser.add_argument("--dry-run", action="store_true", help="List package counts and checksums without writing a ZIP")
    args = parser.parse_args()
    root, output = args.source.resolve(), args.output.resolve()
    if not (root / "backend").is_dir() or not (root / "package.json").is_file():
        parser.error("--source must be the Splat Studio source directory")
    if output.suffix.lower() != ".zip":
        parser.error("--output must have a .zip suffix")
    if output.exists():
        parser.error("Output already exists; use a new filename to preserve the previous package")
    records = []
    for path in source_files(root):
        size = path.stat().st_size
        if size > args.max_file_mib * 1024 ** 2:
            raise ValueError(f"Unexpected large source file: {path.relative_to(root)} ({size} bytes)")
        records.append({"path": path.relative_to(root).as_posix(), "bytes": size, "sha256": sha256(path)})
    records.sort(key=lambda row: row["path"])
    total = sum(row["bytes"] for row in records)
    if total > args.max_total_mib * 1024 ** 2:
        raise ValueError(f"Selected sources exceed {args.max_total_mib} MiB; inspect exclusions before packaging")
    manifest = {"format": 1, "root": "splat-studio", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": records, "total_uncompressed_bytes": total,
        "excluded_directories": sorted(EXCLUDED_DIRECTORIES), "excluded_directory_prefixes": ["release"],
        "excluded_relative_paths": [p.as_posix() for p in EXCLUDED_RELATIVE_PATHS],
        "excluded_suffixes": sorted(EXCLUDED_SUFFIXES),
        "symlinks": "excluded; never followed", "scope": "Source only. No benchmark datasets, checkpoints, model weights, release binaries, or benchmark result trees."}
    if not args.dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".zip.partial")
        try:
            with zipfile.ZipFile(temporary, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for row in records:
                    path = root / row["path"]
                    payload = path.read_bytes()
                    if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
                        raise RuntimeError(f"Source changed during packaging: {row['path']}; rerun after edits finish")
                    archive.writestr("splat-studio/" + row["path"], payload)
                archive.writestr("splat-studio/SOURCE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            temporary.replace(output)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    print(json.dumps({"dry_run": args.dry_run, "files": len(records), "uncompressed_mib": round(total / 1024 ** 2, 2),
        "output": None if args.dry_run else str(output), "archive_sha256": None if args.dry_run else sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
