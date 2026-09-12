"""Read a small, explicit result catalogue without scanning or loading models."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path, PurePosixPath
from urllib.parse import quote

LIBRARY_FORMAT = 'splat-studio-result-library/1'
MAX_REGISTRY_BYTES = 512 * 1024
MAX_RESULTS = 100


def _text(value, limit, *, empty=False):
    if not isinstance(value, str) or len(value) > limit or (not empty and not value.strip()):
        raise ValueError('结果目录文本无效')
    if any(ord(c) < 32 and c not in '\n\t' for c in value):
        raise ValueError('结果目录含有控制字符')
    return value


def _scene_relative(value):
    value = _text(value, 512)
    if any(c in value for c in '\\%?#:') or any(ord(c) < 32 for c in value):
        raise ValueError('结果场景路径无效')
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(p in {'.', '..'} for p in path.parts):
        raise ValueError('结果场景必须位于项目目录中')
    if path.suffix.lower() != '.json':
        raise ValueError('结果场景必须为 JSON')
    return path


def _metrics(value):
    if not isinstance(value, list) or len(value) > 30:
        raise ValueError('结果指标过多')
    out = []
    for metric in value:
        if not isinstance(metric, dict):
            raise ValueError('结果指标无效')
        item = {'label': _text(metric.get('label'), 100)}
        number = metric.get('value')
        if isinstance(number, bool):
            raise ValueError('结果指标值无效')
        if isinstance(number, (int, float)):
            try:
                finite = math.isfinite(number)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError('结果指标值必须有限')
            item['value'] = number
        else:
            item['value'] = _text(number, 300)
        item['scope'] = _text(metric.get('scope', ''), 300, empty=True)
        out.append(item)
    return out


def _result(root, row):
    if not isinstance(row, dict):
        raise ValueError('结果条目无效')
    rid, pid = row.get('id'), row.get('project_id')
    if not isinstance(rid, str) or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,79}', rid):
        raise ValueError('结果标识无效')
    if not isinstance(pid, str) or not re.fullmatch(r'[a-f0-9]{32}', pid):
        raise ValueError('结果项目标识无效')
    relative = _scene_relative(row.get('scene_path'))
    project = (root / pid).resolve()
    # A registry entry cannot make a symlink outside DATA into a registered project.
    if project != root / pid or not project.is_dir():
        raise ValueError('结果项目不存在')
    marker = (project / 'project.json').resolve()
    scene = (project / str(relative)).resolve()
    if not marker.is_relative_to(project) or not marker.is_file():
        raise ValueError('结果项目不存在')
    if not scene.is_relative_to(project) or not scene.is_file():
        raise ValueError('结果场景不存在')
    notes = row.get('notes', [])
    if not isinstance(notes, list) or len(notes) > 20:
        raise ValueError('结果备注过多')
    return {
        'id': rid,
        'title': _text(row.get('title'), 120),
        'description': _text(row.get('description', ''), 1200, empty=True),
        'project_id': pid,
        'scene_url': f'/api/projects/{pid}/files/{quote(str(relative), safe="/")}',
        'metrics': _metrics(row.get('metrics', [])),
        'notes': [_text(note, 600) for note in notes],
    }


def list_saved_results(data):
    """Return only valid available entries; absent catalogue is an empty library.

    Malformed catalogue envelopes are errors. Individual stale or invalid entries
    are omitted, allowing the other explicitly registered results to stay usable.
    No project metadata, scene JSON, chunk data, or Gaussian arrays are read here.
    """
    root = Path(data).resolve()
    registry = root / 'result-library.json'
    if not registry.exists():
        return {'results': []}
    if not registry.resolve().is_relative_to(root) or not registry.is_file():
        raise ValueError('结果目录文件路径无效')
    with registry.open('rb') as handle:
        raw = handle.read(MAX_REGISTRY_BYTES + 1)
    if len(raw) > MAX_REGISTRY_BYTES:
        raise ValueError('结果目录文件过大')
    try:
        catalog = json.loads(raw)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError('结果目录 JSON 无效') from exc
    if not isinstance(catalog, dict) or catalog.get('format') != LIBRARY_FORMAT:
        raise ValueError('结果目录格式不受支持')
    rows = catalog.get('results')
    if not isinstance(rows, list) or len(rows) > MAX_RESULTS:
        raise ValueError('结果目录最多包含 100 项')
    results, seen = [], set()
    for row in rows:
        try:
            item = _result(root, row)
        except (ValueError, OSError, RuntimeError):
            continue
        if item['id'] in seen:
            continue
        seen.add(item['id'])
        results.append(item)
    return {'results': results}
