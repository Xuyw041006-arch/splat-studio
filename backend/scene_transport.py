"""Bounded JSON transport for the viewer; canonical scene.json stays unchanged.

Each chunk preserves exact Gaussian values, order and global source IDs. This is
a transport copy, never resampling. It costs additional disk space approximately
the Gaussian portion of the canonical JSON. It does not reduce browser scene RAM.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

from .scene_limits import MAX_SCENE_LINE_BYTES, MAX_JSONL_IMPORT_BYTES, validate_gaussian_count

MANIFEST_FORMAT = 'splat-studio-scene-chunks/1'
CHUNK_FORMAT = 'splat-studio-scene-chunk/1'
LINES_FORMAT = 'splat-studio-scene-lines/1'


def read_scene_lines(path, *, max_line_bytes=MAX_SCENE_LINE_BYTES):
    """Read our exported JSONL without creating one scene-sized text string.

    Counts, exact row offsets and UTF-8 line byte limits are checked before
    appending each chunk. Gaussian values and metadata are preserved unchanged.
    """
    if type(max_line_bytes) is not int or not 1<=max_line_bytes<=MAX_SCENE_LINE_BYTES:
        raise ValueError('JSONL 单行上限无效')
    path=Path(path)
    if path.stat().st_size>MAX_JSONL_IMPORT_BYTES:raise ValueError('JSONL 超过 4 GiB 文件上限')
    header=None;gaussians=[];total_bytes=0
    with path.open('rb') as stream:
        while True:
            line=stream.readline(max_line_bytes+2)
            if not line:break
            total_bytes+=len(line)
            if total_bytes>MAX_JSONL_IMPORT_BYTES:raise ValueError('JSONL 超过 4 GiB 文件上限')
            if not line.endswith(b'\n'):raise ValueError('JSONL 单行过大或缺少完整结束换行')
            if len(line)-1>max_line_bytes:raise ValueError('JSONL 单行超过 64 MiB 上限')
            try:value=json.loads(line[:-1].decode('utf-8',errors='strict'))
            except (UnicodeError,json.JSONDecodeError) as exc:raise ValueError('JSONL 包含无效 UTF-8 或不完整 JSON') from exc
            if header is None:
                if not isinstance(value,dict) or value.get('format')!=LINES_FORMAT or not isinstance(value.get('scene'),dict) or 'gaussians' in value['scene']:
                    raise ValueError('JSONL 场景格式或头信息无效')
                validate_gaussian_count(value.get('gaussian_count'))
                header=value
                continue
            if not isinstance(value,dict) or value.get('format')!=CHUNK_FORMAT or type(value.get('offset')) is not int or value['offset']!=len(gaussians) or type(value.get('count')) is not int or not 1<=value['count']<=25000 or len(gaussians)+value['count']>header['gaussian_count'] or not isinstance(value.get('gaussians'),list) or len(value['gaussians'])!=value['count'] or not all(isinstance(row,dict) for row in value['gaussians']):
                raise ValueError('JSONL 分块顺序、数量或内容不一致')
            gaussians.extend(value['gaussians'])
    if header is None or len(gaussians)!=header['gaussian_count']:
        raise ValueError('JSONL 文件缺少头信息或完整高斯分块')
    return {**header['scene'],'gaussians':gaussians}


def _write_json(path, data):
    with path.open('w', encoding='utf-8') as stream:
        # Viewer records are already bounded to one header or <=25,000 points.
        stream.write(json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')))


def write_viewer_scene(scene, canonical_path, *, threshold=50_000, chunk_size=25_000):
    """Return canonical_path for small scenes, or a new relative-chunk manifest.

    The caller writes canonical JSON first. A unique chunk folder prevents a new
    publication from mutating chunks a browser is still reading. The manifest is
    replaced only after every chunk is written. Existing transport folders remain
    as run evidence; removing the project/run also removes these copies.
    """
    if not isinstance(scene, dict) or not isinstance(scene.get('gaussians'), list):
        raise ValueError('Viewer transport requires a scene with a Gaussian list')
    validate_gaussian_count(len(scene['gaussians']))
    if type(threshold) is not int or threshold < 0 or type(chunk_size) is not int or not 1 <= chunk_size <= 25_000:
        raise ValueError('Invalid viewer transport threshold or chunk size')
    canonical = Path(canonical_path)
    if len(scene['gaussians']) <= threshold:
        return canonical
    canonical.parent.mkdir(parents=True, exist_ok=True)
    identifier = uuid.uuid4().hex
    folder = canonical.parent/(canonical.stem+'.viewer-data-'+identifier)
    manifest_path = canonical.with_name(canonical.stem+'.viewer.json')
    pending = manifest_path.with_name(manifest_path.name+'.'+identifier+'.partial')
    folder.mkdir()
    try:
        chunks, values = [], scene['gaussians']
        for offset in range(0, len(values), chunk_size):
            count = min(chunk_size, len(values)-offset)
            path = folder/f'chunk-{len(chunks):06d}.json'
            _write_json(path, {'format': CHUNK_FORMAT, 'offset': offset, 'count': count,
                               'gaussians': values[offset:offset+count]})
            chunks.append({'path': path.relative_to(canonical.parent).as_posix(),
                           'offset': offset, 'count': count})
        header = {key: value for key, value in scene.items() if key != 'gaussians'}
        _write_json(pending, {'format': MANIFEST_FORMAT, 'gaussian_count': len(values),
                              'chunk_size': chunk_size, 'chunks': chunks, 'scene': header,
                              'preservation': 'Exact canonical Gaussian values and row order; no transport resampling.'})
        pending.replace(manifest_path)
        return manifest_path
    except BaseException:
        pending.unlink(missing_ok=True)
        shutil.rmtree(folder)
        raise
