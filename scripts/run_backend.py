"""Frozen entry point; the explicit self-test never starts a server or downloads."""
import sys
sys.dont_write_bytecode = True


def geometry_self_test():
    import importlib
    import json
    from backend.learned_geometry import geometry_availability, _paths, verify_source, _runtime
    status = geometry_availability()
    download_capability = geometry_availability({'allow_geometry_download': True})
    _, repo, _ = _paths({})
    verify_source(repo)
    if status['missing_dependencies']:
        raise RuntimeError('Missing frozen geometry dependencies: ' + repr(status['missing_dependencies']))
    _runtime(repo)
    importlib.import_module('dust3r.model')
    importlib.import_module('dust3r.cloud_opt')
    import torch
    import numpy as np
    import roma
    from einops import rearrange
    from scipy.spatial import cKDTree
    assert torch.allclose(roma.rotvec_to_rotmat(torch.zeros(3)), torch.eye(3))
    assert rearrange(torch.zeros(2, 3), 'a b -> b a').shape == (3, 2)
    assert cKDTree(np.zeros((1, 3))).query(np.ones(3))[1] == 0
    print(json.dumps({'status': 'passed', 'frozen': bool(getattr(sys, 'frozen', False)), 'geometry': status,
        'pinned_source_verified': True, 'dust3r_model_cloud_optimizer_imported': True,
        'capability_with_download_authorized': download_capability,
        'torch_numpy_roma_einops_scipy_operations': True, 'checkpoint_loaded': False,
        'downloads_performed': False, 'gpu_inference_tested': False}, ensure_ascii=False))


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test-geometry']:
        geometry_self_test()
    else:
        from backend.app import main
        main()
