"""User-facing names and explicitly approximate training budgets."""
import math
import re


def scene_name(value):
    value = str(value or '').strip()
    if not value or len(value) > 64 or re.search(r'[<>:"/\\|?*\x00-\x1f]', value) or value.endswith('.'):
        raise ValueError('请填写 1–64 个字的场景名，不要包含 / \\ : * ? 等文件名特殊符号。')
    if re.fullmatch(r'(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', value, re.I):
        raise ValueError('这个名称是系统保留名称，请换一个场景名。')
    return value


def estimate_training(analysis, config, hardware=None):
    """An interval for compute only. Never present user-selected GPUs as detected."""
    hardware = hardware or {}
    cloud = config.get('execution_target') == 'cloud'
    if not cloud and not hardware.get('original_3dgs'):
        return {'available': False, 'seconds': None, 'basis': '本机尚未配置可用的 NVIDIA 训练环境。'}
    mode = config.get('mode', 'balanced')
    iterations, resolution, semantic_steps = {'fast': (7000, 4, 400), 'balanced': (15000, 2, 1200), 'fine': (22000, 1, 2400)}[mode]
    n = max(2, analysis.get('unique_image_count', analysis.get('image_count', 2)))
    sizes = [i['width'] * i['height'] for i in analysis.get('sampled_images', []) if i.get('width', 0) > 0 and i.get('height', 0) > 0]
    pixels = sum(sizes) / len(sizes) if sizes else 988 * 730
    workload = max(.5, pixels / (988 * 730)) ** .65 * (2 / resolution) ** 1.2 * max(.8, n / 6) ** .35
    gpu = config.get('cloud_gpu', 'a100') if cloud else hardware.get('gpu_name', 'NVIDIA GPU')
    measured = config.get('measured_steps_per_second')
    if measured is not None:
        if not math.isfinite(measured) or not .1 <= measured <= 10000:
            raise ValueError('实测速度需在 0.1–10000 步/秒之间。')
        rgb = iterations / measured
        basis = '按你提供的同档位实测速率估算。'
        spread = (.7, 1.8)
    elif 'a100' in gpu.lower():
        rgb = 205 * (iterations / 15000) * workload
        basis = '参考 A100 六照片均衡任务的实测耗时，按照片与档位折算。'
        spread = (.65, 2.5)
    else:
        rgb = iterations / 30 * workload
        basis = '该 GPU 尚未标定，暂用通用 GPU 速度粗估；可填写实测速率。'
        spread = (.4, 3.)
    semantics = bool(config.get('semantics'))
    extra = (34 + 19.5 * semantic_steps / 1200 * max(1, n / 6)) if semantics else 0
    if semantics and config.get('priority_request'): extra += .15 * rgb
    if config.get('sparse_completion') != 'none' and (n <= 2 or analysis.get('connected_ratio', 1) < .75):
        extra += 60  # Uncalibrated completion overhead, included in the wide interval.
    base = rgb + extra + max(10, n * 1.5)
    return {'available': True, 'seconds': [max(15, round(base * spread[0])), round(base * spread[1])],
            'basis': basis, 'scope': '仅计算时间；首次安装、模型下载、上传下载另计。',
            'device_label': gpu, 'device_verified': not cloud, 'approximate': True}
