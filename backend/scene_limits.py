"""Shared desktop display resource limits; full means no Gaussian sampling."""

MAX_SCENE_GAUSSIANS = 2_000_000
MAX_PLY_IMPORT_BYTES = 1024 ** 3
MAX_JSON_IMPORT_BYTES = 256 * 1024 ** 2
MAX_JSONL_IMPORT_BYTES = 4 * 1024 ** 3
MAX_SCENE_LINE_BYTES = 64 * 1024 ** 2
PREVIEW_POINT_BUDGETS = {'fast': 100_000, 'balanced': 250_000, 'fine': 500_000}


def import_file_limit(suffix):
    limits={'.ply':MAX_PLY_IMPORT_BYTES,'.json':MAX_JSON_IMPORT_BYTES,'.jsonl':MAX_JSONL_IMPORT_BYTES}
    if suffix not in limits:raise ValueError('仅支持原始 3DGS PLY、场景 JSON 或分块 JSONL')
    return limits[suffix]


def scene_point_budget(quality='full'):
    if quality == 'full':
        return None
    if quality not in PREVIEW_POINT_BUDGETS:
        raise ValueError('显示质量必须为 full、fast、balanced 或 fine')
    return PREVIEW_POINT_BUDGETS[quality]


def validate_gaussian_count(count):
    if type(count) is not int or not 1 <= count <= MAX_SCENE_GAUSSIANS:
        raise ValueError(f'场景需要 1 至 {MAX_SCENE_GAUSSIANS} 个高斯；超出当前桌面资源上限时不会自动抽样，请明确选择预览预算或使用更大容量的渲染器。')
