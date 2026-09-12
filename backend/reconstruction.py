"""Small, real CPU/MPS Gaussian-splat reconstruction reference implementation.

This is deliberately a bounded preview backend, not a claim that the CUDA
submodules in graphdeco-inria/gaussian-splatting can run on Apple Metal.  It uses
the original paper's covariance projection and ordered alpha composition, with
degree-0 spherical harmonics. Full-resolution work belongs to the upstream CUDA
adapter. All estimated cameras use OpenCV's x-right, y-down, z-forward convention.
"""
from __future__ import annotations

import math
import json
import hashlib
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    from .gaussian_io import write_ply, write_scene
except ImportError:
    from gaussian_io import write_ply, write_scene

PRESETS = {
    "fast": {"image_size": 64, "iterations": 48, "max_gaussians": 96},
    "balanced": {"image_size": 96, "iterations": 80, "max_gaussians": 160},
    "fine": {"image_size": 128, "iterations": 120, "max_gaussians": 224},
}


class ReconstructionError(RuntimeError):
    """Actionable failure, never replaced by synthetic successful geometry."""

    def __init__(self, message, code="reconstruction_failed", diagnostics=None):
        super().__init__(message)
        self.code = code
        self.diagnostics = diagnostics or {}


class ReconstructionCancelled(ReconstructionError):
    pass


def _check_cancel(cancelled: Callable[[], bool] | None) -> None:
    if cancelled and cancelled():
        raise ReconstructionCancelled("Reconstruction cancelled")


def _report(progress: Callable | None, value: float, message: str) -> None:
    if progress:
        progress(float(value), message)


def choose_device(requested: str = "auto") -> str:
    import torch
    available = {"cpu": True, "mps": bool(torch.backends.mps.is_available()), "cuda": bool(torch.cuda.is_available())}
    if requested == "auto":
        return "cuda" if available["cuda"] else "mps" if available["mps"] else "cpu"
    if requested not in available:
        raise ReconstructionError(f"Unknown device {requested}")
    if not available[requested]:
        raise ReconstructionError(f"Requested {requested} device is unavailable; select CPU or use the CUDA Colab runner.")
    return requested


def _load_images(image_paths: list[str], max_size: int = 1200) -> list[np.ndarray]:
    from PIL import Image, ImageOps
    images = []
    for path in image_paths:
        with Image.open(path) as opened:
            if opened.width * opened.height > 80_000_000:
                raise ReconstructionError(f"Image is too large: {Path(path).name}")
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((max_size, max_size))
            images.append(np.asarray(image).copy())
    return images


def _intrinsic(image: np.ndarray, focal_ratio: float) -> np.ndarray:
    height, width = image.shape[:2]
    return np.array([[max(height, width) * focal_ratio, 0, width / 2], [0, max(height, width) * focal_ratio, height / 2], [0, 0, 1]], dtype=np.float64)


def _camera(index: int, image: np.ndarray, intrinsic: np.ndarray, transform: np.ndarray, path: str = "") -> dict:
    return {"image_index": index, "image_path": path, "width": int(image.shape[1]), "height": int(image.shape[0]), "intrinsics": intrinsic.tolist(), "world_to_camera": transform.tolist(), "position": (-transform[:3, :3].T @ transform[:3, 3]).tolist()}


def _triangulate(cv2, pixels1, pixels2, k1, k2, transform2):
    normalized1 = cv2.undistortPoints(np.asarray(pixels1, np.float64).reshape(-1, 1, 2), k1, None).reshape(-1, 2)
    normalized2 = cv2.undistortPoints(np.asarray(pixels2, np.float64).reshape(-1, 1, 2), k2, None).reshape(-1, 2)
    homogeneous = cv2.triangulatePoints(np.eye(3, 4), transform2[:3], normalized1.T, normalized2.T)
    xyz = (homogeneous[:3] / np.where(abs(homogeneous[3:4]) > 1e-10, homogeneous[3:4], np.nan)).T
    second = xyz @ transform2[:3, :3].T + transform2[:3, 3]
    ray1 = xyz / np.maximum(np.linalg.norm(xyz, axis=1, keepdims=True), 1e-10)
    center2 = -transform2[:3, :3].T @ transform2[:3, 3]
    ray2 = xyz - center2
    ray2 /= np.maximum(np.linalg.norm(ray2, axis=1, keepdims=True), 1e-10)
    angle = np.degrees(np.arccos(np.clip((ray1 * ray2).sum(1), -1, 1)))
    projection1 = xyz @ k1.T
    projection2 = second @ k2.T
    projection1 = projection1[:, :2] / projection1[:, 2:3]
    projection2 = projection2[:, :2] / projection2[:, 2:3]
    error = 0.5 * (np.linalg.norm(projection1 - pixels1, axis=1) + np.linalg.norm(projection2 - pixels2, axis=1))
    valid = np.isfinite(xyz).all(1) & (xyz[:, 2] > 0.05) & (second[:, 2] > 0.05) & (angle > 1.0) & (error < 3.0) & (np.linalg.norm(xyz, axis=1) < 200)
    return xyz, valid, error, angle


def build_sparse_scene(image_paths: list[str], config: dict | None = None, progress: Callable | None = None, cancelled: Callable | None = None) -> dict:
    """Recover geometry from these input images, add tracks, and run bounded BA.

    No external full-scene poses or points are read. Intrinsics are approximate
    unless the caller supplies calibrated matrices in resized-feature pixels.
    """
    import cv2
    from .sparse_geometry import image_coverage, extend_registered_geometry, bundle_adjust
    config = config or {}
    diagnostic = {"backend": "sfm", "input_views": len(image_paths), "pairs": [],
                  "thresholds": {"mutual_matches": 16, "triangulated_points": 12, "minimum_parallax_degrees": 1., "maximum_reprojection_px": 3.},
                  "warning_codes": [], "external_pose_or_pointcloud_used": False}
    def fail(code, message):
        diagnostic.update(status="failed", failure_code=code)
        raise ReconstructionError(message, code, diagnostic)
    if len(image_paths) < 2:
        fail("insufficient_views", "At least two overlapping views are required. A single image cannot provide measured 3D geometry.")
    if len(image_paths) > 80:
        fail("preview_capacity", "Local preview supports at most 80 images; use the upstream CUDA/COLMAP workflow for larger scenes.")
    images = _load_images(image_paths, int(config.get("feature_image_size", 1200)))
    _check_cancel(cancelled)
    fingerprints = [hashlib.sha256(str(im.shape).encode()+im.tobytes()).hexdigest() for im in images]
    seen = {}; duplicates = []
    for i, fingerprint in enumerate(fingerprints):
        if fingerprint in seen: duplicates.append(i)
        else: seen[fingerprint] = i
    diagnostic["duplicate_view_indices"] = duplicates
    diagnostic["input_manifest"] = []
    for i, path in enumerate(image_paths):
        with Path(path).open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        diagnostic["input_manifest"].append({"index": i, "name": Path(path).name, "sha256": digest,
                                             "feature_width": images[i].shape[1], "feature_height": images[i].shape[0]})
    if len(seen) < 2:
        fail("duplicate_views", "No reliable stereo geometry: duplicate photos do not provide distinct viewpoints.")
    focal_ratio = float(config.get("focal_ratio", 0.9))
    if not 0.2 <= focal_ratio <= 5: fail("invalid_intrinsics", "focal_ratio must lie between 0.2 and 5")
    intrinsics = [_intrinsic(im, focal_ratio) for im in images]
    supplied = config.get("intrinsics")
    if config.get("intrinsics_original") is not None:
        if supplied is not None: fail("invalid_intrinsics", "Supply either intrinsics_original or resized intrinsics, not both")
        original = config['intrinsics_original']; sizes = config.get('original_sizes')
        if not isinstance(sizes,(list,tuple)) or len(original)!=len(images) or len(sizes)!=len(images):
            fail("invalid_intrinsics", "Original intrinsics require one [width,height] original_sizes entry per input")
        supplied = []
        from PIL import Image, ImageOps
        for k, size, image, path in zip(original,sizes,images,image_paths):
            k=np.asarray(k,np.float64).copy(); size=np.asarray(size,np.float64)
            if k.shape!=(3,3) or size.shape!=(2,) or not np.isfinite(size).all() or (size<=0).any():
                fail("invalid_intrinsics", "Invalid original camera dimensions or intrinsic matrix")
            with Image.open(path) as opened: actual_size=ImageOps.exif_transpose(opened).size
            if tuple(size)!=actual_size:
                fail("invalid_intrinsics", "original_sizes must match each actual orientation-corrected input image")
            k[0] *= image.shape[1]/size[0]; k[1] *= image.shape[0]/size[1]; supplied.append(k)
    if supplied is not None:
        if len(supplied) != len(images): fail("invalid_intrinsics", "One intrinsic matrix is required per resized feature image")
        intrinsics = [np.asarray(k, np.float64) for k in supplied]
        if any(k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2],[0,0,1]) for k in intrinsics):
            fail("invalid_intrinsics", "Invalid intrinsic matrices")
    cv2.setRNGSeed(int(config.get("geometry_seed", 7)))
    detector = cv2.SIFT_create(nfeatures=3500); features = []
    for i, im in enumerate(images):
        _check_cancel(cancelled)
        kp, desc = detector.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), None)
        features.append((kp, desc))
        _report(progress, 0.04 + 0.12 * (i + 1) / len(images), f"Extracted features from {i + 1}/{len(images)} views")
    diagnostic["feature_counts"] = [len(item[0]) for item in features]
    matcher = cv2.BFMatcher(cv2.NORM_L2); cache = {}
    def matches_between(i, j):
        key = (i, j)
        if key in cache: return cache[key]
        d1, d2 = features[i][1], features[j][1]
        if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2: cache[key] = []; return []
        forward = [pair[0] for pair in matcher.knnMatch(d1, d2, k=2) if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance]
        backward = {pair[0].queryIdx: pair[0].trainIdx for pair in matcher.knnMatch(d2, d1, k=2) if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance}
        cache[key] = [match for match in forward if backward.get(match.trainIdx) == match.queryIdx]
        return cache[key]
    candidates = list(dict.fromkeys([(i, i + 1) for i in range(len(images) - 1)] + [(0, j) for j in range(2, len(images))] + [(i, min(len(images) - 1, i + max(2, len(images) // 3))) for i in range(1, len(images) - 2)]))[:160]
    best = None; best_score = 0
    for pair_no, (i, j) in enumerate(candidates):
        _check_cancel(cancelled)
        if i in duplicates or j in duplicates: continue
        matches = matches_between(i, j)
        row = {"pair": [i,j], "mutual_matches": len(matches), "status": "rejected", "reason": "insufficient_matches"}
        diagnostic["pairs"].append(row)
        if len(matches) < 16: continue
        p1 = np.array([features[i][0][m.queryIdx].pt for m in matches])
        p2 = np.array([features[j][0][m.trainIdx].pt for m in matches])
        row["matching_coverage"] = [image_coverage(p1,images[i].shape), image_coverage(p2,images[j].shape)]
        _, hmask = cv2.findHomography(p1,p2,cv2.RANSAC,3.)
        row["homography_inliers"] = int(hmask.sum()) if hmask is not None else 0
        n1 = cv2.undistortPoints(p1.reshape(-1, 1, 2), intrinsics[i], None).reshape(-1, 2)
        n2 = cv2.undistortPoints(p2.reshape(-1, 1, 2), intrinsics[j], None).reshape(-1, 2)
        threshold = 1.5 / max(intrinsics[i][0, 0], intrinsics[j][0, 0])
        essential, mask = cv2.findEssentialMat(n1, n2, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=threshold)
        if essential is None or essential.shape != (3, 3): row["reason"] = "essential_estimation_failed"; continue
        _, rotation, translation, pose_mask = cv2.recoverPose(essential, n1, n2, np.eye(3), mask=mask)
        transform = np.eye(4); transform[:3, :3], transform[:3, 3] = rotation, translation[:, 0]
        xyz, valid, error, angle = _triangulate(cv2, p1, p2, intrinsics[i], intrinsics[j], transform)
        valid &= pose_mask[:, 0] > 0
        finite_angles = angle[np.isfinite(angle)]
        row.update(pose_inliers=int((pose_mask>0).sum()), triangulated_valid=int(valid.sum()),
                   median_candidate_parallax_deg=float(np.median(finite_angles)) if len(finite_angles) else None)
        if valid.sum() < 12:
            row["reason"] = "insufficient_parallax" if len(finite_angles) and np.median(finite_angles) <= 1. else "inconsistent_triangulation"
            continue
        row.update(status="usable",reason=None,median_reprojection_error_px=float(np.median(error[valid])),
                   median_parallax_deg=float(np.median(angle[valid])),coverage=[image_coverage(p1[valid],images[i].shape),image_coverage(p2[valid],images[j].shape)])
        score = int(valid.sum()) * min(float(np.median(angle[valid])), 12)
        if score > best_score:
            best_score = score; best = (i, j, matches, xyz, valid, error, angle, transform, p1, p2, row)
        _report(progress, 0.16 + 0.13 * (pair_no + 1) / len(candidates), f"Matched view pair {pair_no + 1}/{len(candidates)}")
    if best is None:
        if max(diagnostic["feature_counts"], default=0) < 16: code = "insufficient_features"
        elif max((p["mutual_matches"] for p in diagnostic["pairs"]), default=0) < 16: code = "insufficient_overlap"
        elif any(p.get('reason') == 'insufficient_parallax' for p in diagnostic['pairs']): code = "insufficient_parallax"
        else: code = "inconsistent_geometry"
        fail(code, "No reliable stereo geometry was found (need ≥12 triangulated matches with parallax). Photograph textured surfaces with overlap and translation; pure rotation, identical photos and featureless objects cannot be reconstructed by this preview.")
    first, second, matches, all_xyz, valid, error, angle, transform, p1, p2, seed_diagnostic = best
    xyz = all_xyz[valid]; matches = [m for m, keep in zip(matches, valid) if keep]
    pixels, pixels2 = p1[valid], p2[valid]
    sampled1 = images[first][np.clip(np.rint(pixels[:, 1]).astype(int),0,images[first].shape[0]-1),np.clip(np.rint(pixels[:, 0]).astype(int),0,images[first].shape[1]-1)]/255.
    sampled2 = images[second][np.clip(np.rint(pixels2[:, 1]).astype(int),0,images[second].shape[0]-1),np.clip(np.rint(pixels2[:, 0]).astype(int),0,images[second].shape[1]-1)]/255.
    cameras = [_camera(first, images[first], intrinsics[first], np.eye(4), image_paths[first]), _camera(second, images[second], intrinsics[second], transform, image_paths[second])]
    track_map = {first: {m.queryIdx: idx for idx,m in enumerate(matches)},second:{m.trainIdx:idx for idx,m in enumerate(matches)}}
    observations = [{str(first):a.tolist(),str(second):b.tolist()} for a,b in zip(pixels,pixels2)]
    confidence = np.clip(np.minimum(angle[valid]/8,1)*np.exp(-error[valid]/3),.05,1)
    expanded = extend_registered_geometry(xyz,(sampled1+sampled2)/2,confidence,observations,cameras,track_map,
        images,image_paths,intrinsics,features,matches_between,_triangulate,_camera,duplicates,config,
        lambda:_check_cancel(cancelled),progress)
    xyz,cameras,ba = bundle_adjust(expanded["points"],expanded["cameras"],expanded["observations"],{first,second},config,lambda:_check_cancel(cancelled))
    rejected = expanded["rejected_views"]
    diagnostic.update(status="partial" if rejected else "success", seed_pair=[first,second],seed_point_count=len(all_xyz[valid]),
                      registration=expanded["diagnostics"],bundle_adjustment=ba)
    warnings = []
    intrinsic_sources=config.get('intrinsics_sources') or (['supplied_unverified']*len(images) if supplied is not None else ['heuristic']*len(images))
    if supplied is None:
        warnings.append(f"Camera intrinsics use a {focal_ratio:g}×image-size focal prior; metric scale and calibration are not recovered.")
    elif any(source!='user_calibrated' for source in intrinsic_sources):
        warnings.append('Some camera intrinsics use heuristic, EXIF-based, or unverified input estimates; they are not independently calibrated and metric scale is not recovered.')
    if duplicates: diagnostic["warning_codes"].append("duplicate_views_excluded"); warnings.append(f"{len(duplicates)} duplicate views were excluded; duplicate images do not add geometry evidence.")
    if rejected: diagnostic["warning_codes"].append("partial_registration"); warnings.append(f"{len(rejected)} views could not be independently registered and are excluded from training.")
    if min(c["hull_fraction"] for c in seed_diagnostic["coverage"]) < .10:
        diagnostic["warning_codes"].append("localized_support"); warnings.append("Verified geometry covers a small image region; the rest of the scene remains unsupported.")
    if any(c["border_fraction"] > .1 for c in seed_diagnostic["coverage"]): diagnostic["warning_codes"].append("border_support_partial_observation")
    if len(cameras) <= 2:
        diagnostic["warning_codes"].append("unobserved_surfaces_unknown"); warnings.append("The SfM stage registered only two views; unseen surfaces remain unknown. Any later learned-visible completion is reported separately.")
    warnings.append("Bounded SfM uses verified multi-view triangulation and fixed-gauge bundle adjustment; it is not full COLMAP mapping or unseen-surface completion.")
    metadata = {"registered_views":len(cameras),"input_views":len(images),"rejected_view_indices":rejected,"seed_pair":[first,second],
        "seed_triangulated_points":len(all_xyz[valid]),"additional_triangulated_points":expanded['diagnostics']['added_points'],
        "triangulated_points":len(xyz),"median_reprojection_error_px":float(np.median(error[valid])),
        "median_triangulation_angle_deg":float(np.median(angle[valid])),"seed_diagnostics_scope":"seed pair before bundle adjustment",
        "camera_source":"estimated_from_supplied_images_only","geometry_source":"verified_multiview_triangulation", "scale":"arbitrary; seed baseline = 1",
        "intrinsics_sources":intrinsic_sources,"geometry_diagnostics":diagnostic,"warnings":warnings}
    return {"points":xyz.astype(np.float32),"colors":expanded['colors'].astype(np.float32),"confidence":expanded['confidence'].astype(np.float32),
            "observations":expanded['observations'],"cameras":cameras,"images":images,"sources":["observed"]*len(xyz),"metadata":metadata}


def _validated_learned_geometry(result, image_paths):
    """Verify the learned/observed boundary before either trainer consumes it."""
    if not isinstance(result,dict): raise ReconstructionError('Learned geometry returned no scene','invalid_learned_geometry')
    points=np.asarray(result.get('points'),np.float32);colors=np.asarray(result.get('colors'),np.float32)
    confidence=np.asarray(result.get('confidence'),np.float32)
    if points.ndim!=2 or points.shape[1]!=3 or not len(points) or colors.shape!=points.shape or confidence.shape!=(len(points),):
        raise ReconstructionError('Learned geometry arrays have incompatible shapes','invalid_learned_geometry')
    if not np.isfinite(points).all() or not np.isfinite(colors).all() or not np.isfinite(confidence).all() or (colors<0).any() or (colors>1).any() or (confidence<0).any() or (confidence>.49+1e-6).any():
        raise ReconstructionError('Learned geometry contains invalid values or unverified confidence','invalid_learned_geometry')
    if result.get('sources')!=['inferred']*len(points):
        raise ReconstructionError('Every learned point must be marked inferred','invalid_learned_provenance')
    images=result.get('images');cameras=result.get('cameras');observations=result.get('observations')
    if not isinstance(images,list) or len(images)!=len(image_paths) or not isinstance(cameras,list) or len(cameras)<2 or not isinstance(observations,list) or len(observations)!=len(points):
        raise ReconstructionError('Learned geometry needs input images, camera identities, and per-point observations','invalid_learned_geometry')
    used={}
    for camera in cameras:
        index=camera.get('image_index')
        if not isinstance(index,int) or isinstance(index,bool) or not 0<=index<len(images) or index in used:
            raise ReconstructionError('Learned camera does not uniquely identify an input image','invalid_learned_geometry')
        image=np.asarray(images[index]);k=np.asarray(camera.get('intrinsics'),np.float64);t=np.asarray(camera.get('world_to_camera'),np.float64)
        if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8 or (camera.get('height'),camera.get('width'))!=image.shape[:2]:
            raise ReconstructionError('Learned image size and camera pixel coordinates disagree','invalid_learned_geometry')
        if k.shape!=(3,3) or t.shape!=(4,4) or not np.isfinite(k).all() or not np.isfinite(t).all() or k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],[0,0,1]) or not np.allclose(t[3],[0,0,0,1]) or not np.allclose(t[:3,:3]@t[:3,:3].T,np.eye(3),atol=.01) or np.linalg.det(t[:3,:3])<.99:
            raise ReconstructionError('Learned camera requires finite pinhole intrinsics and a rigid transform','invalid_learned_geometry')
        if Path(camera.get('image_path','')).name!=Path(image_paths[index]).name:
            raise ReconstructionError('Learned camera filename is not its declared input image','invalid_learned_geometry')
        used[str(index)]=camera
    for track in observations:
        if not isinstance(track,dict) or not track:raise ReconstructionError('Learned points need actual image support','invalid_learned_geometry')
        for frame,pixel in track.items():
            xy=np.asarray(pixel,np.float64);camera=used.get(frame)
            if camera is None or xy.shape!=(2,) or not np.isfinite(xy).all() or not 0<=xy[0]<camera['width'] or not 0<=xy[1]<camera['height']:
                raise ReconstructionError('Learned point observation is outside its input pixel grid','invalid_learned_geometry')
    result.update(points=points,colors=colors,confidence=confidence)
    result.setdefault('metadata',{}).setdefault('warnings',[])
    return result


def _learned_geometry_availability(config):
    try:
        from .learned_geometry import geometry_availability
        result=geometry_availability(config)
        if result.get('requires_download') and not config.get('allow_geometry_download',False):
            return {**result,'available':False,'reason':'geometry_model_download_not_authorized'}
        return result
    except ImportError as exc:
        return {'available':False,'requires_download':False,'reason':'learned_geometry_module_unavailable','detail':str(exc)}


def initialize_geometry(image_paths, config=None, progress=None, cancelled=None):
    """Shared photo-only initialization and explicit learned-visible completion.

    auto preserves a valid SfM seed. An available model may add aligned inferred
    points for extreme two-view input; failed SfM may instead use an explicitly
    identified learned initialization. Model downloads are never implied.
    """
    config=dict(config or {})
    backend=config.get('geometry_backend','auto');completion=config.get('sparse_completion','auto')
    if backend not in {'auto','sfm','dust3r'}:raise ReconstructionError('Unknown geometry_backend','invalid_geometry_backend')
    if completion not in {'auto','none','learned_visible'}:raise ReconstructionError('Unknown sparse_completion','invalid_completion_mode')
    def learned(cfg):
        from .learned_geometry import initialize_learned_geometry
        return _validated_learned_geometry(initialize_learned_geometry(image_paths,cfg,progress,cancelled),image_paths)
    if backend=='dust3r':
        availability=_learned_geometry_availability(config)
        if not availability.get('available'):
            raise ReconstructionError('Learned initialization unavailable: '+str(availability.get('reason')),'learned_geometry_unavailable',availability)
        result=learned(config)
        result['metadata'].update(geometry_backend_requested=backend,geometry_backend_applied='dust3r',
            sparse_completion={'requested':completion,'status':'learned_visible_initialization','unseen_surfaces_recovered':False})
        return result
    try:
        result=build_sparse_scene(image_paths,config,progress,cancelled)
    except ReconstructionCancelled:raise
    except ReconstructionError as error:
        if backend=='sfm' or completion=='none' or error.code in {'insufficient_views','duplicate_views','invalid_intrinsics','preview_capacity'}:raise
        availability=_learned_geometry_availability(config)
        if not availability.get('available'):
            error.diagnostics['learned_fallback']={'status':'unavailable',**availability}
            raise
        _check_cancel(cancelled)
        try:result=learned(config)
        except ReconstructionCancelled:raise
        except Exception as learned_error:
            error.diagnostics['learned_fallback']={'status':'failed','error':str(learned_error)}
            raise ReconstructionError(str(error)+' Learned fallback also failed: '+str(learned_error),error.code,error.diagnostics) from learned_error
        result['metadata'].update(geometry_backend_requested=backend,geometry_backend_applied='dust3r',
            sfm_failure={'code':error.code,'diagnostics':error.diagnostics},
            sparse_completion={'requested':completion,'status':'learned_visible_initialization_after_sfm_failure','unseen_surfaces_recovered':False})
        return result
    result['metadata'].update(geometry_backend_requested=backend,geometry_backend_applied='sfm')
    triggers=[]
    if len(image_paths)==2:triggers.append('two_view_input')
    if len(image_paths)<=6 and len(result['cameras'])<len(image_paths):triggers.append('partial_registration')
    if len(image_paths)<=6 and len(result['points'])<64:triggers.append('few_verified_points')
    requested=completion=='learned_visible' or (completion=='auto' and bool(triggers))
    if not requested:
        result['metadata']['sparse_completion']={'requested':completion,'status':'disabled' if completion=='none' else 'not_needed_for_input_view_count','unseen_surfaces_recovered':False}
        return result
    availability=_learned_geometry_availability(config)
    if not availability.get('available'):
        result['metadata']['sparse_completion']={'requested':completion,'status':'unavailable','triggers':triggers,'availability':availability,'unseen_surfaces_recovered':False}
        result['metadata']['warnings'].append('Visible-region completion is unavailable; only the verified observed geometry is returned. '+str(availability.get('reason')))
        return result
    from PIL import Image, ImageOps
    sizes=[]
    for path in image_paths:
        with Image.open(path) as image:sizes.append(ImageOps.exif_transpose(image).size)
    by_frame={str(c['image_index']):c for c in result['cameras']}
    samples=[{frame:(np.asarray(pixel)*[sizes[int(frame)][0]/by_frame[frame]['width'],sizes[int(frame)][1]/by_frame[frame]['height']]).tolist()
              for frame,pixel in track.items()} for track in result['observations']]
    cfg={**config,'geometry_alignment_observations':samples}
    try:
        _check_cancel(cancelled)
        prediction=learned(cfg)
        from .sparse_geometry import align_visible_completion
        result,diagnostic=align_visible_completion(result,prediction,config,lambda:_check_cancel(cancelled))
        diagnostic['learned_initializer']={key:prediction.get('metadata',{}).get(key)
            for key in ('geometry_source','camera_source','geometry_diagnostics','warnings')}
    except ReconstructionCancelled:raise
    except Exception as error:
        diagnostic={'status':'failed_preserved_observed','reason':str(error),'provider_error_code':getattr(error,'code',None),
            'provider_error_diagnostics':getattr(error,'diagnostics',{}),'observed_geometry_preserved':True,'unseen_surfaces_recovered':False,'added_points':0}
    result['metadata']['sparse_completion']={'requested':completion,'triggers':triggers,**diagnostic}
    if diagnostic['status']!='completed_visible_only':
        result['metadata']['warnings'].append('Learned visible completion was not accepted; verified observed geometry was preserved. '+str(diagnostic.get('reason')))
    return result


def build_imported_scene(image_paths: list[str], imported: dict | str, config: dict, cancelled=None) -> dict:
    """Use externally supplied geometry/cameras, preserving prediction provenance.

    This is a bootstrap route for actual learned outputs or calibrated datasets,
    not a camera or depth generator. Camera estimates are not independently verified.
    """
    _check_cancel(cancelled)
    if not image_paths:
        raise ReconstructionError("Imported-scene training requires its corresponding source images")
    if isinstance(imported, (str, Path)):
        path = Path(imported)
        if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
            raise ReconstructionError("Imported scene JSON is missing or exceeds the 32MB preview limit")
        imported = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(imported, dict) or not isinstance(imported.get("gaussians"), list) or not imported["gaussians"] or len(imported["gaussians"]) > 100_000:
        raise ReconstructionError("Imported scene must contain 1–100000 Gaussians")
    if not isinstance(imported.get("cameras"), list) or not imported["cameras"]:
        raise ReconstructionError("Imported scene must include calibrated or externally estimated cameras")
    images = _load_images(image_paths, int(config.get("feature_image_size", 1200)))
    points, colors, scales, rotations, opacities, confidence, sources, observations = [], [], [], [], [], [], [], []
    for gaussian in imported["gaussians"]:
        arrays = {}
        for key, shape in [("position", (3,)), ("color", (3,)), ("scale", (3,)), ("rotation", (4,))]:
            try:
                value = np.asarray(gaussian[key], dtype=np.float32)
            except (KeyError, TypeError, ValueError) as exc:
                raise ReconstructionError(f"Imported Gaussian has invalid {key}") from exc
            if value.shape != shape or not np.isfinite(value).all():
                raise ReconstructionError(f"Imported Gaussian has invalid {key}")
            arrays[key] = value
        opacity = float(gaussian.get("opacity", .6))
        score = float(gaussian.get("confidence", .25))
        if (arrays["scale"] <= 0).any() or (arrays["color"] < 0).any() or (arrays["color"] > 1).any() or not 0 <= opacity <= 1 or not math.isfinite(score):
            raise ReconstructionError("Imported Gaussian has invalid scale/color/opacity/confidence")
        norm = np.linalg.norm(arrays["rotation"])
        if norm < 1e-6:
            raise ReconstructionError("Imported Gaussian has a zero quaternion")
        source = "observed" if gaussian.get("source") == "observed" else "inferred"
        points.append(arrays["position"]); colors.append(arrays["color"]); scales.append(arrays["scale"])
        rotations.append(arrays["rotation"] / norm); opacities.append(opacity)
        confidence.append(float(np.clip(score, 0, 1 if source == "observed" else .49)))
        sources.append(source); observations.append(gaussian.get("observations", {}))
    cameras, used = [], set()
    for supplied in imported["cameras"]:
        if "image_index" in supplied:
            index = int(supplied["image_index"])
        else:
            name = Path(supplied.get("image_path", supplied.get("image_name", ""))).name
            matches = [i for i, path in enumerate(image_paths) if Path(path).name == name]
            if len(matches) != 1:
                raise ReconstructionError("Each imported camera must uniquely match an input image_index or image filename")
            index = matches[0]
        if index < 0 or index >= len(images) or index in used:
            raise ReconstructionError("Imported camera image indices must be unique and refer to provided images")
        used.add(index)
        intrinsic = supplied.get("intrinsics", supplied.get("K"))
        if isinstance(intrinsic, dict):
            intrinsic = [[intrinsic["fx"], 0, intrinsic["cx"]], [0, intrinsic.get("fy", intrinsic["fx"]), intrinsic["cy"]], [0, 0, 1]]
        k, transform = np.asarray(intrinsic, np.float64), np.asarray(supplied.get("world_to_camera"), np.float64)
        if k.shape != (3, 3) or transform.shape != (4, 4) or not np.isfinite(k).all() or not np.isfinite(transform).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
            raise ReconstructionError("Imported cameras need finite 3×3 intrinsics and 4×4 world_to_camera matrices")
        rotation = transform[:3, :3]
        if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-5) or not np.allclose(rotation @ rotation.T, np.eye(3), atol=.01) or np.linalg.det(rotation) < .99:
            raise ReconstructionError("Imported camera transforms must be rigid OpenCV world-to-camera matrices")
        width, height = float(supplied.get("width", 0)), float(supplied.get("height", 0))
        if not math.isfinite(width+height) or width <= 0 or height <= 0:
            raise ReconstructionError("Imported cameras must state the image width/height used for their intrinsics")
        k[0] *= images[index].shape[1] / width; k[1] *= images[index].shape[0] / height
        cameras.append(_camera(index, images[index], k, transform, image_paths[index]))
    return {"points": np.asarray(points, np.float32), "colors": np.asarray(colors, np.float32), "scales": scales, "rotations": rotations, "opacities": opacities, "confidence": np.asarray(confidence, np.float32), "sources": sources, "observations": observations, "cameras": cameras, "images": images, "metadata": {"registered_views": len(cameras), "input_views": len(images), "rejected_view_indices": sorted(set(range(len(images)))-used), "triangulated_points": 0, "imported_points": len(points), "camera_source": "externally_supplied", "geometry_source": "imported_scene", "scale": imported.get("metadata", {}).get("scale", "external coordinate scale"), "warnings": ["Geometry and cameras were imported, not recovered by this app. Prediction provenance is preserved; observation claims and calibration are not independently verified.", "Only registered source images are used for photometric refinement. Missing surfaces require an actual configured completion provider."]}}


def quaternion_matrix(quaternions):
    import torch
    q = quaternions / quaternions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y), 2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x), 2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)], dim=-1).reshape(-1, 3, 3)


def render_gaussians(positions, log_scales, quaternions, color_logits, opacity_logits, intrinsic, world_to_camera, height: int, width: int, background=None):
    """Differentiable full-covariance projection and front-to-back alpha splatting.

    O(N×H×W) memory, meant for <256 splats at <=128px. No CUDA extensions.
    """
    import torch
    device, dtype = positions.device, positions.dtype
    rotation = quaternion_matrix(quaternions)
    cov3d = (rotation * torch.exp(2 * log_scales).unsqueeze(1)) @ rotation.transpose(1, 2)
    camera_rotation = world_to_camera[:3, :3]
    camera_xyz = positions @ camera_rotation.T + world_to_camera[:3, 3]
    depth = camera_xyz[:, 2]
    z = depth.clamp_min(0.05)
    x, y = camera_xyz[:, 0], camera_xyz[:, 1]
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    center = torch.stack([fx * x / z + intrinsic[0, 2], fy * y / z + intrinsic[1, 2]], dim=-1)
    zero = torch.zeros_like(z)
    jacobian = torch.stack([fx / z, zero, -fx * x / z.square(), zero, fy / z, -fy * y / z.square()], -1).reshape(-1, 2, 3)
    camera_cov = camera_rotation.unsqueeze(0) @ cov3d @ camera_rotation.T.unsqueeze(0)
    cov2d = jacobian @ camera_cov @ jacobian.transpose(1, 2) + torch.eye(2, device=device, dtype=dtype).unsqueeze(0) * 0.3
    # Analytic inverse avoids LAPACK/MPS inverse support differences.
    a, b, c = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
    determinant = (a * c - b.square()).clamp_min(1e-8)
    grid_y, grid_x = torch.meshgrid(torch.arange(height, device=device, dtype=dtype), torch.arange(width, device=device, dtype=dtype), indexing="ij")
    dx = grid_x.unsqueeze(0) - center[:, 0, None, None]
    dy = grid_y.unsqueeze(0) - center[:, 1, None, None]
    power = -(c[:, None, None] * dx.square() - 2 * b[:, None, None] * dx * dy + a[:, None, None] * dy.square()) / (2 * determinant[:, None, None])
    alpha = (opacity_logits.sigmoid().reshape(-1, 1, 1) * torch.exp(power.clamp(max=0)) * (depth > 0.05)[:, None, None]).clamp(max=0.99)
    order = torch.argsort(depth)
    alpha = alpha[order]
    colors = color_logits.sigmoid()[order]
    transmission = torch.cumprod(torch.cat([torch.ones((1, height, width), device=device, dtype=dtype), 1 - alpha + 1e-8], dim=0), dim=0)
    weights = alpha * transmission[:-1]
    output = torch.einsum("nhw,nc->hwc", weights, colors)
    if background is not None:
        output = output + transmission[-1, :, :, None] * background
    return output


def _priority_weight_map(config: dict, camera: dict, height: int, width: int) -> np.ndarray:
    """Import user-confirmed 2D priority masks; never infer semantic labels here."""
    import cv2
    from PIL import Image
    masks = config.get("priority_masks") or {}
    image_path = camera.get("image_path", "")
    entries = []
    if isinstance(masks, dict):
        for key in [image_path, Path(image_path).name, str(camera["image_index"])]:
            if key and key in masks:
                value = masks[key]
                entries = value if isinstance(value, list) and value and isinstance(value[0], dict) else [value]
                break
    elif isinstance(masks, list):
        entries = [entry for entry in masks if isinstance(entry, dict) and entry.get("image_path", entry.get("image_name")) in (image_path, Path(image_path).name)]
    weights = np.ones((height, width), dtype=np.float32)
    strength = float(np.clip(config.get("priority_strength", 3), 0, 10))
    for entry in entries:
        value = entry.get("mask_path", entry.get("mask")) if isinstance(entry, dict) else entry
        if value is None:
            continue
        if isinstance(value, (str, Path)):
            with Image.open(value) as opened:
                mask = np.asarray(opened.convert("L"), dtype=np.float32)
        else:
            mask = np.asarray(value, dtype=np.float32)
        if mask.ndim == 3:
            mask = mask.max(axis=2)
        if mask.ndim != 2 or not np.isfinite(mask).all():
            raise ReconstructionError("Priority masks must be finite 2D arrays or image paths")
        if mask.max(initial=0) > 1:
            mask = mask / 255.0
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST).clip(0, 1)
        weights = np.maximum(weights, 1 + strength * mask)
    return weights


def train_gaussians(points: np.ndarray, colors: np.ndarray, cameras: list[dict], images: list[np.ndarray], config: dict | None = None, progress: Callable | None = None, cancelled: Callable | None = None) -> dict:
    """Optimize positions, anisotropic scales, rotation, SH DC and opacity."""
    import torch
    import cv2
    config = config or {}
    if config.get("semantic_strategy", config.get("semantic_mode")) in ("joint", "joint_loss"):
        raise ReconstructionError("Joint semantic-loss training is not implemented in the bounded reference backend; select posthoc semantic embedding. Priority-mask weighted RGB optimization is supported.")
    preset = PRESETS.get(config.get("mode", "balanced"), PRESETS["balanced"])
    image_size = max(16, min(128, int(config.get("image_size", preset["image_size"]))))
    iterations = max(1, min(250, int(config.get("iterations", preset["iterations"]))))
    max_gaussians = max(1, min(256, int(config.get("max_gaussians", preset["max_gaussians"]))))
    device_name = choose_device(config.get("device", "auto"))
    device = torch.device(device_name)
    if device_name == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    if len(points) == 0 or len(cameras) == 0:
        raise ReconstructionError("Training requires triangulated points and registered cameras")
    sparse_training = bool(config.get('sparse', False))
    few_views = len(cameras) <= 2
    anchor_strength = (.04 if few_views else .02) if sparse_training else .005
    movement_bound = (.05 if few_views else .08) if sparse_training else .15
    position_learning_rate = (.00075 if few_views else .001) if sparse_training else .0015
    scale_anchor_strength = .005 if sparse_training else 0.
    point_sources = config.get('initial_sources', ['observed']*len(points))
    if not isinstance(point_sources,(list,tuple)) or len(point_sources)!=len(points) or any(s not in {'observed','inferred'} for s in point_sources):
        raise ReconstructionError('initial_sources must match every initialization point','invalid_geometry_provenance')
    _check_cancel(cancelled)
    # Reproducible sampling, preserving an index back to feature tracks. Imported
    # priority masks also allocate more of the bounded Gaussian budget locally.
    if len(points) > max_gaussians:
        sampling_weight = np.ones(len(points), dtype=np.float64)
        if config.get("priority_masks"):
            for camera in cameras:
                k = np.asarray(camera["intrinsics"], dtype=np.float64)
                transform = np.asarray(camera["world_to_camera"], dtype=np.float64)
                camera_points = points @ transform[:3, :3].T + transform[:3, 3]
                projected = camera_points @ k.T
                uv = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-8)
                ratio = 64 / max(camera["width"], camera["height"])
                w, h = max(1, round(camera["width"] * ratio)), max(1, round(camera["height"] * ratio))
                u = np.rint(uv[:, 0] * w / camera["width"]).astype(int)
                v = np.rint(uv[:, 1] * h / camera["height"]).astype(int)
                valid = (camera_points[:, 2] > .05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
                weights = _priority_weight_map(config, camera, h, w)
                sampling_weight[valid] = np.maximum(sampling_weight[valid], weights[v[valid], u[valid]])
        probabilities = sampling_weight / sampling_weight.sum() if config.get("priority_masks") else None
        rng=np.random.default_rng(7)
        observed=np.flatnonzero(np.asarray(point_sources)=='observed')
        if sparse_training and 0<len(observed)<len(points):
            reserve=min(len(observed),max(1,max_gaussians//2))
            observed_weights=sampling_weight[observed];observed_weights=observed_weights/observed_weights.sum()
            preserved=rng.choice(observed,reserve,replace=False,p=observed_weights)
            remaining=np.setdiff1d(np.arange(len(points)),preserved)
            remaining_weights=sampling_weight[remaining];remaining_weights=remaining_weights/remaining_weights.sum()
            indices=np.r_[preserved,rng.choice(remaining,max_gaussians-reserve,replace=False,p=remaining_weights)]
        else:
            indices = rng.choice(len(points), max_gaussians, replace=False, p=probabilities)
    else:
        indices = np.arange(len(points))
    chosen_points, chosen_colors = points[indices], colors[indices]
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    positions = torch.nn.Parameter(tensor(chosen_points.copy()))
    original_positions = positions.detach().clone()
    selected_sources=[point_sources[int(i)] for i in indices]
    anchor_weights=tensor([1. if source=='observed' else .25 for source in selected_sources])[:,None]
    extent = max(float(np.linalg.norm(np.std(chosen_points, axis=0))), 0.05)
    if len(chosen_points) > 1:
        distances = np.linalg.norm(chosen_points[:, None, :] - chosen_points[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        scales = np.clip(np.min(distances, axis=1) * 0.35, extent * 0.005, extent * 0.3)
    else:
        scales = np.array([extent * 0.2])
    if config.get("initial_scales") is not None:
        initial = np.asarray(config["initial_scales"], dtype=np.float32)
        scales3 = initial[indices] if initial.ndim == 2 else np.repeat(initial[indices, None], 3, axis=1)
    else:
        scales3 = np.repeat(scales[:, None], 3, axis=1)
    log_scales = torch.nn.Parameter(tensor(np.log(np.maximum(scales3, 1e-6))))
    original_log_scales = log_scales.detach().clone()
    initial_rotations = np.asarray(config["initial_rotations"], np.float32)[indices] if config.get("initial_rotations") is not None else np.tile([1, 0, 0, 0], (len(indices), 1))
    quaternions = torch.nn.Parameter(tensor(initial_rotations))
    colors_clipped = np.clip(chosen_colors, 0.03, 0.97)
    color_logits = torch.nn.Parameter(tensor(np.log(colors_clipped / (1 - colors_clipped))))
    initial_opacity = np.clip(np.asarray(config["initial_opacities"], np.float32)[indices], .001, .999) if config.get("initial_opacities") is not None else np.full(len(indices), .6, np.float32)
    opacity_logits = torch.nn.Parameter(tensor(np.log(initial_opacity / (1 - initial_opacity))))
    optimizer = torch.optim.Adam([{"params": [positions], "lr": position_learning_rate * extent}, {"params": [log_scales], "lr": 0.008}, {"params": [quaternions], "lr": 0.004}, {"params": [color_logits], "lr": 0.045}, {"params": [opacity_logits], "lr": 0.018}])
    train_views = []
    for camera in cameras:
        image = images[camera["image_index"]]
        ratio = min(image_size / image.shape[1], image_size / image.shape[0])
        width, height = max(8, round(image.shape[1] * ratio)), max(8, round(image.shape[0] * ratio))
        target = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
        if image.dtype == np.uint8:
            target /= 255.0
        k = np.asarray(camera["intrinsics"], dtype=np.float32).copy()
        k[0] *= width / camera["width"]
        k[1] *= height / camera["height"]
        train_views.append((tensor(target), tensor(k), tensor(camera["world_to_camera"]), height, width, tensor(_priority_weight_map(config, camera, height, width))))
    # Fixed per-scene background is documented in the output; no invented geometry.
    background = tensor(config.get("background", [0.05, 0.05, 0.05]))
    losses = []
    elapsed_steps = []

    def render_view(view):
        target, intrinsic, transform, height, width, _ = view
        return render_gaussians(positions, log_scales, quaternions, color_logits, opacity_logits, intrinsic, transform, height, width, background)

    with torch.no_grad():
        initial_loss = float(torch.stack([(render_view(v) - v[0]).abs().mean() for v in train_views]).mean().cpu())
    started = time.perf_counter()
    view_rng = np.random.default_rng(int(config.get('geometry_seed',7)))
    view_order = np.arange(len(train_views)); view_visits = np.zeros(len(train_views),dtype=int)
    for step in range(iterations):
        _check_cancel(cancelled)
        step_started = time.perf_counter()
        if sparse_training and step % len(train_views) == 0: view_order = view_rng.permutation(len(train_views))
        view_index = int(view_order[step % len(train_views)])
        view = train_views[view_index]; view_visits[view_index] += 1
        optimizer.zero_grad(set_to_none=True)
        rendered = render_view(view)
        photometric = ((rendered - view[0]).abs() * view[5][:, :, None]).sum() / (3 * view[5].sum())
        # Anchor movement to measured points; insufficient views must not imply completion.
        regularizer = anchor_strength * (((positions - original_positions) / extent).square()*anchor_weights).mean()
        regularizer = regularizer + scale_anchor_strength * (log_scales-original_log_scales).square().mean()
        loss = photometric + regularizer
        if not torch.isfinite(loss):
            raise ReconstructionError("Training produced non-finite loss; inspect calibration and geometry")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            log_scales.clamp_(math.log(extent * 0.002), math.log(extent * 0.6))
            positions.copy_(original_positions + (positions - original_positions).clamp(-extent * movement_bound, extent * movement_bound))
            opacity_logits.clamp_(-5, 5)
        # A scalar CPU transfer synchronizes MPS/CUDA, making measured ETA meaningful.
        losses.append(float(photometric.detach().cpu()))
        elapsed_steps.append(time.perf_counter() - step_started)
        if step % 5 == 0 or step == iterations - 1:
            remaining = float(np.mean(elapsed_steps[-10:])) * (iterations - step - 1)
            _report(progress, 0.40 + 0.50 * (step + 1) / iterations, f"Optimizing splats {step + 1}/{iterations} · L1 {losses[-1]:.4f} · measured ETA {remaining:.0f}s")
    with torch.no_grad():
        final_loss = float(torch.stack([(render_view(v) - v[0]).abs().mean() for v in train_views]).mean().cpu())
        arrays = [p.detach().cpu().numpy() for p in [positions, log_scales.exp(), quaternions / quaternions.norm(dim=-1, keepdim=True).clamp_min(1e-8), color_logits.sigmoid(), opacity_logits.sigmoid()]]
    gaussians = [{"position": arrays[0][i].tolist(), "scale": arrays[1][i].tolist(), "rotation": arrays[2][i].tolist(), "color": arrays[3][i].tolist(), "opacity": float(arrays[4][i]), "confidence": 0.5, "source": "observed", "track_index": int(indices[i])} for i in range(len(indices))]
    return {"gaussians": gaussians, "selected_indices": indices.tolist(), "metrics": {"device": device_name, "iterations": iterations, "image_size": image_size, "gaussian_count": len(gaussians), "initial_l1": initial_loss, "final_l1": final_loss, "training_seconds": time.perf_counter() - started, "median_step_seconds": float(np.median(elapsed_steps)), "loss_history": losses, "sh_degree": 0, "background": background.cpu().tolist(), "priority_weighted": bool(config.get("priority_masks")),
        "effective_regularization":{"sparse":sparse_training,"few_registered_views":few_views,"position_anchor_weight":anchor_strength,
            "scale_anchor_weight":scale_anchor_strength,"position_bound_extent_fraction":movement_bound,"position_learning_rate_extent_factor":position_learning_rate,
            "inferred_position_anchor_multiplier":.25,"selected_observed_points":selected_sources.count('observed'),"selected_inferred_points":selected_sources.count('inferred'),
            "view_sampling":"shuffled_balanced_epochs" if sparse_training else "cyclic_balanced","sampling_seed":int(config.get('geometry_seed',7)),
            "view_visit_counts":{str(c['image_index']):int(n) for c,n in zip(cameras,view_visits)}},
        "loss_definition": "initial/final: unweighted mean training-view RGB L1; history: priority-weighted RGB L1, all at preview resolution, not novel-view quality"}}


def _apply_completion_prior(scene: dict, config: dict) -> None:
    strategy = config.get("completion", "none")
    if strategy in (None, False, "none", "auto"):
        scene["metadata"]["completion"] = {"method": "none", "status": "unseen geometry remains unknown", "learned_provider_interface": "Provide callable completion_provider(scene, config); returned Gaussians are marked inferred."}
        return
    if strategy == "learned":
        provider = config.get("completion_provider")
        if not callable(provider):
            raise ReconstructionError("Learned completion was requested but no completion_provider is configured. No completed geometry has been fabricated.")
        inferred = provider(scene, config)
        if not isinstance(inferred, list) or len(inferred) > 4096:
            raise ReconstructionError("Completion provider must return a bounded list of at most 4096 Gaussians")
        for gaussian in inferred:
            if not isinstance(gaussian, dict):
                raise ReconstructionError("Completion provider entries must be Gaussian objects")
            for name, shape in [("position", (3,)), ("scale", (3,)), ("rotation", (4,)), ("color", (3,)), ("opacity", ())]:
                if name not in gaussian:
                    raise ReconstructionError(f"Invalid completion Gaussian field {name}")
                value = np.asarray(gaussian[name], dtype=np.float32)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ReconstructionError(f"Invalid completion Gaussian field {name}")
            if min(gaussian["scale"]) <= 0 or min(gaussian["color"]) < 0 or max(gaussian["color"]) > 1 or not 0 <= gaussian["opacity"] <= 1:
                raise ReconstructionError("Completion scales must be positive; RGB and opacity must lie in [0, 1]")
            q = np.asarray(gaussian["rotation"], dtype=np.float32)
            if np.linalg.norm(q) < 1e-6:
                raise ReconstructionError("Completion quaternion must have nonzero norm")
            gaussian["rotation"] = (q / np.linalg.norm(q)).tolist()
            gaussian["source"] = "inferred"
            gaussian["confidence"] = float(np.clip(gaussian.get("confidence", 0.25), 0, 0.49))
        scene["gaussians"].extend(inferred)
        scene["metadata"]["completion"] = {"method": "learned_provider", "count": len(inferred), "status": "inferred geometry; not measured"}
        return
    if strategy != "bounded_prior":
        raise ReconstructionError(f"Unknown completion strategy {strategy}")
    # Conservative local normal-thickness prior only. It cannot create unseen objects.
    inferred = []
    for gaussian in scene["gaussians"][:256]:
        duplicate = dict(gaussian)
        duplicate["scale"] = [min(s * 1.15, s + 0.02) for s in gaussian["scale"]]
        duplicate["opacity"] = min(0.12, gaussian["opacity"] * 0.15)
        duplicate["source"] = "inferred"
        duplicate["confidence"] = 0.15
        duplicate.pop("track_index", None)
        inferred.append(duplicate)
    scene["gaussians"].extend(inferred)
    scene["metadata"]["completion"] = {"method": "bounded_local_support_prior", "count": len(inferred), "status": "weak local support expansion only; does not recover unseen surfaces", "max_scale_growth": 1.15}
    scene["metadata"]["warnings"].append("Optional inferred splats only expand local Gaussian support by ≤15%; this is a geometric prior, not learned or measured completion.")


def reconstruct(image_paths: list[str], output_dir: str, config: dict | None = None, progress: Callable | None = None, cancelled: Callable | None = None) -> dict:
    config = config or {}
    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if config.get("mode", "balanced") not in PRESETS:
        raise ReconstructionError("mode must be fast, balanced or fine")
    choose_device(config.get("device", "auto"))
    if config.get("completion") == "learned" and not callable(config.get("completion_provider")):
        raise ReconstructionError("Learned completion requires a configured provider. Choose none or the explicitly labeled bounded geometric prior.")
    _report(progress, 0.01, "Analyzing image geometry")
    imported = config.get("imported_scene", config.get("imported_scene_path"))
    if imported is not None:
        sparse = build_imported_scene(image_paths, imported, config, cancelled)
        train_config = {**config, "initial_scales": sparse["scales"], "initial_rotations": sparse["rotations"], "initial_opacities": sparse["opacities"], "initial_sources":sparse['sources']}
        _report(progress, .36, "Validated imported geometry and cameras; preserving inferred provenance")
    else:
        sparse = initialize_geometry(image_paths, config, progress, cancelled)
        train_config = {**config,'initial_sources':sparse.get('sources',['observed']*len(sparse['points']))}
    trained = train_gaussians(sparse["points"], sparse["colors"], sparse["cameras"], sparse["images"], train_config, progress, cancelled)
    for gaussian, track_index in zip(trained["gaussians"], trained["selected_indices"]):
        gaussian["confidence"] = float(sparse["confidence"][track_index])
        gaussian["observations"] = sparse["observations"][track_index]
        if sparse.get("sources") is not None:
            gaussian["source"] = sparse["sources"][track_index]
    metadata = {**sparse["metadata"], **trained["metrics"], "backend": "torch_reference_preview", "mode": config.get("mode", "balanced"), "strategy": "sparse_two_view" if len(sparse["cameras"]) <= 2 else "sparse_multiview", "semantic_requested": bool(config.get("semantic", False)), "semantic_status": "pending external semantic stage" if config.get("semantic") else "disabled", "coordinate_convention": "OpenCV x-right y-down z-forward; world-to-camera matrices", "limitations": ["Bounded preview only: <=256 observed splats and <=128px training images.", "Degree-0 SH; no view-dependent appearance, SSIM, densification or pruning.", "Approximate camera intrinsics unless calibrated matrices are supplied.", "No guarantee of unseen-view fidelity; training loss is not a quality certificate."]}
    scene = {"version": 1, "gaussians": trained["gaussians"], "cameras": sparse["cameras"], "metadata": metadata}
    if imported is not None:
        metadata["strategy"] = "imported_geometry_refinement"
    _apply_completion_prior(scene, config)
    _check_cancel(cancelled)
    scene["metadata"]["total_seconds"] = time.perf_counter() - started
    scene["metadata"]["gaussian_count"] = len(scene["gaussians"])
    if metadata["final_l1"] >= metadata["initial_l1"]:
        metadata["warnings"].append("Training did not improve mean photometric loss. Treat this reconstruction as an unsuccessful fit.")
    _report(progress, 0.94, "Saving Gaussian scene and original-format PLY")
    scene_path = write_scene(scene, output / "scene.json")
    ply_path = write_ply(scene, output / "point_cloud.ply")
    _report(progress, 1.0, "Reconstruction preview finished")
    return {"scene_path": scene_path, "ply_path": ply_path, "metadata": scene["metadata"], "scene": scene}
