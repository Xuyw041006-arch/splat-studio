"""Semantic compute budgets, independent of the RGB iteration budget.

These are engineering presets, not measured accuracy/speed rankings. A sweep
covers every observed frame/class pair; unseen pairs never become negatives.
"""
from __future__ import annotations

PROFILES = {
    'fast': dict(semantic_steps=400, train_size=192, minimum_sweeps=1,
                 semantic_view_mode='sampled', semantic_max_views=12, semantic_granularity='multilevel'),
    'balanced': dict(semantic_steps=1200, train_size=256, minimum_sweeps=2,
                     semantic_view_mode='sampled', semantic_max_views=24, semantic_granularity='multilevel'),
    'fine': dict(semantic_steps=2400, train_size=384, minimum_sweeps=3,
                 semantic_view_mode='all', semantic_max_views=24, semantic_granularity='multilevel'),
}


def resolve_semantic_profile(config):
    cfg = dict(config)
    mode = cfg.get('mode', 'balanced')
    if mode not in PROFILES:
        raise ValueError('Unknown semantic quality mode')
    budget = cfg.get('semantic_budget', 'manual')
    if budget not in {'mode', 'manual'}:
        raise ValueError('semantic_budget must be mode or manual')
    profile = dict(PROFILES[mode])
    if budget == 'mode':
        for key in ('semantic_steps', 'train_size', 'minimum_sweeps'):
            cfg[key] = profile[key]
        cfg['semantic_max_views'] = profile['semantic_max_views']
    for key in ('semantic_view_mode', 'semantic_granularity'):
        if cfg.get(key, 'auto') == 'auto':
            cfg[key] = profile[key]
        elif cfg[key] not in ({'all', 'sampled'} if key == 'semantic_view_mode' else {'flat', 'multilevel'}):
            raise ValueError('Invalid ' + key)
    cfg.setdefault('semantic_max_views', profile['semantic_max_views'])
    cfg.setdefault('sampling_schedule', 'view_cycle')
    cfg.setdefault('minimum_sweeps', 0)
    cfg.setdefault('train_size', 256)
    cfg.setdefault('semantic_steps', 1200)
    cfg['cross_view_confidence'] = True
    cfg['automatic_hierarchy'] = cfg['semantic_granularity']=='multilevel'
    cfg['semantic_profile'] = dict(mode=mode, budget=budget,
        **{k: cfg[k] for k in profile}, sampling_schedule=cfg['sampling_schedule'],
        accuracy_validated=False,
        timing_scope='RGB, teacher, matching, optimization and export are recorded separately; no fixed GPU ETA')
    return cfg


def worker_semantic_config(config):
    """Keep worker options consistent across initial reconstruction and resume UI."""
    cfg = resolve_semantic_profile(config)
    keys = ('semantic_steps', 'train_size', 'minimum_sweeps', 'sampling_schedule',
            'semantic_profile', 'semantic_granularity', 'granularity_config',
            'approved_hierarchy', 'training_frames', 'held_out_frames',
            'automatic_hierarchy', 'cross_view_confidence')
    return {key: cfg[key] for key in keys if key in cfg}
