"""Bounded geometric diagnostics and gauge-fixed bundle adjustment.

All observations are pixels from supplied input views. This module never loads
external poses, a full-scene point cloud, or generated views.
"""
from __future__ import annotations
import copy
import time
import numpy as np


def image_coverage(pixels, image_shape):
    """Report support distribution; a border observation remains usable."""
    import cv2
    h, w = image_shape[:2]
    xy = np.asarray(pixels, np.float64).reshape(-1, 2)
    valid = np.isfinite(xy).all(1) & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    xy = xy[valid]
    if not len(xy):
        return {"count": 0, "grid_fraction": 0., "hull_fraction": 0., "border_fraction": 0.}
    cells = np.minimum((xy / [w, h] * 8).astype(int), 7)
    area = cv2.contourArea(cv2.convexHull(xy.astype(np.float32))) if len(xy) >= 3 else 0.
    border = (xy[:, 0] < .02*w) | (xy[:, 0] >= .98*w) | (xy[:, 1] < .02*h) | (xy[:, 1] >= .98*h)
    return {"count": len(xy), "grid_fraction": float(len(np.unique(cells, axis=0)) / 64),
            "hull_fraction": float(area / (w*h)), "border_fraction": float(border.mean())}


def project_points(points, camera):
    transform = np.asarray(camera["world_to_camera"], np.float64)
    xyz = np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]
    uv = xyz @ np.asarray(camera["intrinsics"], np.float64).T
    return uv[:, :2] / np.maximum(xyz[:, 2:3], .00001), xyz[:, 2]


def triangulate_registered(pixels1, pixels2, camera1, camera2, triangulator):
    """Triangulate arbitrary registered views, retaining original seed gates."""
    import cv2
    first = np.asarray(camera1["world_to_camera"], np.float64)
    second = np.asarray(camera2["world_to_camera"], np.float64)
    relative = second @ np.linalg.inv(first)
    local, valid, error, angle = triangulator(cv2, pixels1, pixels2,
        np.asarray(camera1["intrinsics"]), np.asarray(camera2["intrinsics"]), relative)
    world = (local - first[:3, 3]) @ first[:3, :3]
    return world, valid, error, angle


def extend_registered_geometry(points, colors, confidence, observations, cameras, track_map,
                               images, image_paths, intrinsics, features, matches_between,
                               triangulator, camera_factory, excluded=(), config=None,
                               check_cancel=None, progress=None):
    """PnP-register views, add verified tracks, then retry previously disconnected views."""
    import cv2
    config = config or {}
    xyz = [p.copy() for p in np.asarray(points)]
    colors = [c.copy() for c in np.asarray(colors)]
    confidence = list(np.asarray(confidence, float))
    observations = copy.deepcopy(observations)
    cameras = copy.deepcopy(cameras)
    registered = {c["image_index"]: c for c in cameras}
    pending = set(range(len(images)))-set(registered)-set(excluded)
    maximum = max(len(xyz), min(10000, max(24, int(config.get('sfm_max_points', 4000)))))
    attempts = []; additions = []; last_reasons = {}
    def add_observation(point_id, frame, feature):
        pixel = np.asarray(features[frame][0][feature].pt)
        predicted, depth = project_points(np.asarray([xyz[point_id]]), registered[frame])
        if depth[0] > .05 and np.linalg.norm(predicted[0]-pixel) < 3.:
            observations[point_id][str(frame)] = pixel.tolist()
            track_map[frame][feature] = point_id
    def correspondences(frame):
        votes = {}
        for anchor in registered:
            for match in matches_between(anchor, frame):
                point_id = track_map[anchor].get(match.queryIdx)
                if point_id is not None: votes.setdefault(point_id, set()).add(match.trainIdx)
        # Neither one point->multiple pixels nor one pixel->multiple points may
        # enter PnP as if they were independent trustworthy observations.
        unique = {pi: next(iter(ids)) for pi, ids in votes.items() if len(ids) == 1}
        reverse = {}
        for pi, feature in unique.items(): reverse.setdefault(feature, []).append(pi)
        return {pi: feature for pi, feature in unique.items() if len(reverse[feature]) == 1}
    while pending:
        made_progress = False
        ordered = sorted(((frame, correspondences(frame)) for frame in pending), key=lambda v: (-len(v[1]), v[0]))
        for frame, correspond in ordered:
            if check_cancel: check_cancel()
            diagnostic = {"image_index": frame, "correspondences": len(correspond)}
            if len(correspond) < 8:
                diagnostic.update(status='rejected', reason='insufficient_2d3d_matches'); attempts.append(diagnostic)
                last_reasons[frame] = diagnostic['reason']; continue
            indices = list(correspond)
            pixels = np.asarray([features[frame][0][correspond[pi]].pt for pi in indices], np.float64)
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(np.asarray(xyz)[indices], pixels, intrinsics[frame], None,
                iterationsCount=200, reprojectionError=4., confidence=.999, flags=cv2.SOLVEPNP_EPNP)
            if not ok or inliers is None or len(inliers) < 8:
                diagnostic.update(status='rejected', reason='pnp_inconsistent'); attempts.append(diagnostic)
                last_reasons[frame] = diagnostic['reason']; continue
            transform = np.eye(4); transform[:3, :3] = cv2.Rodrigues(rvec)[0]; transform[:3, 3] = tvec[:, 0]
            camera = camera_factory(frame, images[frame], intrinsics[frame], transform, image_paths[frame])
            projected, depth = project_points(np.asarray(xyz)[indices], camera)
            error = np.linalg.norm(projected-pixels, axis=1)
            keep = np.zeros(len(indices), bool); keep[inliers[:, 0]] = True
            keep &= (depth > .05) & (error < 4.)
            if keep.sum() < 8:
                diagnostic.update(status='rejected', reason='pnp_cheirality_or_reprojection'); attempts.append(diagnostic)
                last_reasons[frame] = diagnostic['reason']; continue
            registered[frame] = camera; cameras.append(camera); track_map[frame] = {}
            for local in np.flatnonzero(keep):
                pi = indices[local]; track_map[frame][correspond[pi]] = pi
                observations[pi][str(frame)] = pixels[local].tolist()
            diagnostic.update(status='registered', inliers=int(keep.sum()), median_reprojection_error_px=float(np.median(error[keep])),
                              coverage=image_coverage(pixels[keep], images[frame].shape))
            attempts.append(diagnostic); pending.remove(frame); made_progress = True
            # A registered view can contribute new geometry beyond the seed
            # pair and supply tracks for a previously unregistrable next view.
            for anchor in [i for i in registered if i != frame][-24:]:
                if check_cancel: check_cancel()
                candidates = []
                for match in matches_between(anchor, frame):
                    old_a, old_b = track_map[anchor].get(match.queryIdx), track_map[frame].get(match.trainIdx)
                    if old_a is not None and old_b is None: add_observation(old_a, frame, match.trainIdx)
                    elif old_b is not None and old_a is None: add_observation(old_b, anchor, match.queryIdx)
                    elif old_a is None and old_b is None: candidates.append(match)
                if not candidates or len(xyz) >= maximum: continue
                pa = np.asarray([features[anchor][0][m.queryIdx].pt for m in candidates])
                pb = np.asarray([features[frame][0][m.trainIdx].pt for m in candidates])
                added, valid, error, angle = triangulate_registered(pa, pb, registered[anchor], camera, triangulator)
                count = 0
                for local in np.flatnonzero(valid):
                    if len(xyz) >= maximum: break
                    match = candidates[local]
                    if match.queryIdx in track_map[anchor] or match.trainIdx in track_map[frame]: continue
                    pi = len(xyz); xyz.append(added[local]); count += 1
                    track_map[anchor][match.queryIdx] = pi; track_map[frame][match.trainIdx] = pi
                    observations.append({str(anchor): pa[local].tolist(), str(frame): pb[local].tolist()})
                    rgb = []
                    for view, pixel in ((anchor, pa[local]), (frame, pb[local])):
                        h, w = images[view].shape[:2]; x, y = np.rint(pixel).astype(int)
                        rgb.append(images[view][np.clip(y,0,h-1),np.clip(x,0,w-1)]/255.)
                    colors.append(np.mean(rgb, axis=0))
                    confidence.append(float(np.clip(min(angle[local]/8,1)*np.exp(-error[local]/3),.05,1)))
                additions.append({'pair':[anchor,frame], 'untracked_matches':len(candidates), 'added_points':count})
            if progress: progress(.29+.07*len(registered)/len(images),f"Registered {len(registered)}/{len(images)} views; triangulated {len(xyz)} points")
        if not made_progress: break
    return {"points":np.asarray(xyz,np.float64), "colors":np.asarray(colors,np.float64), "confidence":np.asarray(confidence,np.float64),
            "observations":observations, "cameras":cameras, "rejected_views":sorted(pending|set(excluded)),
            "diagnostics":{"attempts":attempts,"new_point_pairs":additions,"added_points":len(xyz)-len(points),
                           "registered_ratio":len(cameras)/len(images),"unregistered_reasons":{str(i):last_reasons.get(i,'duplicate_input') for i in pending|set(excluded)},
                           "point_budget":maximum}}


def bundle_adjust(points, cameras, observations, fixed_view_ids, config=None, check_cancel=None):
    """Refine points and non-seed poses with fixed intrinsics and fixed gauge.

    Both seed poses are fixed, preserving baseline scale and the trusted seed
    coordinate frame. Parameter bounds, anchor residuals, and an all-track
    post-check prevent a numerical optimizer from silently invalidating tracks.
    This is sparse least-squares BA, not dense completion or full COLMAP BA.
    """
    config = config or {}
    initial = np.asarray(points, np.float64)
    original_cameras = copy.deepcopy(cameras)
    diagnostic = {"method": "bounded_gauge_fixed_bundle_adjustment", "fixed_view_ids": sorted(fixed_view_ids),
                  "intrinsics_optimized": False, "accepted": False}
    if not config.get("bundle_adjustment", True):
        return initial.copy(), original_cameras, {**diagnostic, "status": "disabled"}
    try:
        import cv2
        from scipy.optimize import least_squares
        from scipy.sparse import lil_matrix
    except ImportError:
        return initial.copy(), original_cameras, {**diagnostic, "status": "scipy_unavailable"}
    camera_lookup = {str(c["image_index"]): i for i, c in enumerate(cameras)}
    rows = [(pi, camera_lookup[key], xy) for pi, track in enumerate(observations)
            for key, xy in track.items() if key in camera_lookup]
    if not rows:
        return initial.copy(), original_cameras, {**diagnostic, "status": "no_observations"}
    point_ids = np.array([r[0] for r in rows]); camera_ids = np.array([r[1] for r in rows])
    targets = np.asarray([r[2] for r in rows], np.float64)
    # Keep all observations, even when only a bounded subset of points moves.
    limit = max(12, min(1600, int(config.get("ba_max_points", 600))))
    ranked = sorted(range(len(initial)), key=lambda i: (-len(observations[i]), i))
    selected = np.asarray(ranked[:limit], int)
    point_slots = {int(pi): j for j, pi in enumerate(selected)}
    movable = [i for i, c in enumerate(cameras) if c["image_index"] not in fixed_view_ids]
    camera_slots = {ci: len(selected)*3 + j*6 for j, ci in enumerate(movable)}
    dimension = len(selected)*3 + len(movable)*6
    extent = max(float(np.linalg.norm(np.std(initial, axis=0))), .05)
    base_transforms = np.asarray([c["world_to_camera"] for c in cameras], np.float64)
    intrinsics = np.asarray([c["intrinsics"] for c in cameras], np.float64)
    def unpack(parameters):
        xyz = initial.copy(); xyz[selected] += parameters[:len(selected)*3].reshape(-1, 3)*extent
        transforms = base_transforms.copy()
        for ci, slot in camera_slots.items():
            transforms[ci, :3, :3] = cv2.Rodrigues(parameters[slot:slot+3])[0] @ base_transforms[ci, :3, :3]
            transforms[ci, :3, 3] += parameters[slot+3:slot+6]*extent
        return xyz, transforms
    def data_residual(parameters):
        xyz, transforms = unpack(parameters)
        rotated = np.einsum('nij,nj->ni', transforms[camera_ids, :3, :3], xyz[point_ids]) + transforms[camera_ids, :3, 3]
        homogeneous = np.einsum('nij,nj->ni', intrinsics[camera_ids], rotated)
        uv = homogeneous[:, :2] / np.maximum(rotated[:, 2:3], .05)
        return np.column_stack([uv-targets, np.maximum(.05-rotated[:, 2], 0)*100/extent]), rotated[:, 2]
    track_rows=[[] for _ in initial]
    for oi,pi in enumerate(point_ids):track_rows[pi].append(oi)
    def valid_track_geometry(parameters, errors, depth):
        xyz,transforms=unpack(parameters)
        centers=-np.einsum('nji,nj->ni',transforms[:,:3,:3],transforms[:,:3,3])
        supported=np.zeros(len(initial),bool)
        for pi, rows_for_point in enumerate(track_rows):
            for left,oi in enumerate(rows_for_point):
                for oj in rows_for_point[left+1:]:
                    if depth[oi]<=.05 or depth[oj]<=.05 or (errors[oi]+errors[oj])*.5>=3.:continue
                    first=xyz[pi]-centers[camera_ids[oi]];second=xyz[pi]-centers[camera_ids[oj]]
                    norms=np.linalg.norm(first)*np.linalg.norm(second)
                    cosine=np.dot(first,second)/max(norms,1e-12)
                    if np.degrees(np.arccos(np.clip(cosine,-1,1)))>1.:
                        supported[pi]=True;break
                if supported[pi]:break
        return supported
    def residual(parameters):
        if check_cancel: check_cancel()
        data, _ = data_residual(parameters)
        # Small normalized anchors complement strict geometric parameter bounds.
        return np.r_[data.ravel(), parameters*.1]
    sparsity = lil_matrix((len(rows)*3+dimension, dimension), dtype=np.int8)
    for oi, (pi, ci, _) in enumerate(rows):
        if pi in point_slots:
            slot = point_slots[pi]*3; sparsity[oi*3:oi*3+3, slot:slot+3] = 1
        if ci in camera_slots:
            slot = camera_slots[ci]; sparsity[oi*3:oi*3+3, slot:slot+6] = 1
    sparsity[len(rows)*3:, :] = np.eye(dimension, dtype=np.int8)
    bounds = np.full(dimension, .10)
    for slot in camera_slots.values(): bounds[slot:slot+3] = .04; bounds[slot+3:slot+6] = .05
    zero = np.zeros(dimension); before, before_depth = data_residual(zero)
    started = time.perf_counter()
    max_nfev=max(2,min(80,int(config.get('ba_max_nfev',25))))
    result = least_squares(residual, zero, jac_sparsity=sparsity.tocsr(), method='trf', loss='huber', f_scale=1.,
        bounds=(-bounds, bounds), max_nfev=max_nfev,
        ftol=1e-5, xtol=1e-5, gtol=1e-5)
    after, depth = data_residual(result.x)
    before_error = np.linalg.norm(before[:, :2], axis=1); after_error = np.linalg.norm(after[:, :2], axis=1)
    support_before = np.bincount(point_ids[before_error < 4.], minlength=len(initial))
    support_after = np.bincount(point_ids[(after_error < 4.) & (depth > .05)], minlength=len(initial))
    before_geometric=valid_track_geometry(zero,before_error,before_depth)
    after_geometric=valid_track_geometry(result.x,after_error,depth)
    retained = bool(np.all(support_after[support_before >= 2] >= 2) and np.all(after_geometric[before_geometric]))
    improved = np.mean(np.minimum(after_error, 4.)**2) <= np.mean(np.minimum(before_error, 4.)**2)
    accepted = bool(np.isfinite(result.x).all() and np.isfinite(after).all() and retained and improved)
    diagnostic.update(status="accepted" if accepted else "rejected_preserved_original", accepted=accepted,
        seconds=time.perf_counter()-started, optimized_point_count=len(selected), optimized_camera_count=len(movable),
        observation_count=len(rows), function_evaluations=int(result.nfev), max_nfev=max_nfev,
        reprojection_median_before_px=float(np.median(before_error)), reprojection_median_after_px=float(np.median(after_error)),
        all_previously_supported_tracks_retained=retained, minimum_parallax_degrees=1.,maximum_pair_mean_reprojection_px=3.,position_bound_extent_fraction=.10,
        camera_translation_bound_extent_fraction=.05, camera_rotation_bound_radians=.04)
    if not accepted: return initial.copy(), original_cameras, diagnostic
    xyz, transforms = unpack(result.x)
    for camera, transform in zip(original_cameras, transforms):
        camera["world_to_camera"] = transform.tolist()
        camera["position"] = (-transform[:3, :3].T @ transform[:3, 3]).tolist()
    return xyz, original_cameras, diagnostic


def align_visible_completion(sfm, learned, config=None, check_cancel=None):
    """Add only two-view-supported learned points in an unchanged SfM frame.

    Alignment uses sampled learned point maps at SfM observation pixels. A
    deterministic held-out subset checks the fitted similarity transform.
    Neither known SfM positions nor its cameras are replaced by model output.
    """
    import cv2
    from scipy.spatial import cKDTree
    config = config or {}
    diagnostic = {"method":"learned_visible_points_aligned_to_sfm", "status":"rejected",
                  "observed_geometry_preserved":True,"unseen_surfaces_recovered":False,"added_points":0}
    def rejected(reason): return sfm, {**diagnostic,"reason":reason}
    points = np.asarray(sfm['points'],np.float64)
    groups = {}
    for sample in learned.get('alignment_samples',[]):
        pi=sample.get('track_index'); frame=sample.get('frame_index'); value=np.asarray(sample.get('point'),np.float64)
        if not isinstance(pi,int) or not 0<=pi<len(points) or str(frame) not in sfm['observations'][pi]: continue
        if value.shape!=(3,) or not np.isfinite(value).all(): continue
        groups.setdefault(pi,{})[frame]=value
    ids=sorted(pi for pi, samples in groups.items() if len(samples)>=2)
    if len(ids)<16: return rejected('insufficient_shared_alignment_tracks')
    source=np.asarray([np.mean(list(groups[pi].values()),axis=0) for pi in ids]); target=points[ids]
    extent=max(float(np.linalg.norm(np.std(target,axis=0))),.05)
    threshold=.04*extent
    def similarity(a,b):
        ac=a-a.mean(0);bc=b-b.mean(0)
        singular=np.linalg.svd(ac,compute_uv=False)
        if len(singular)<2 or singular[1]<max(singular[0]*1e-4,1e-8): raise ValueError('collinear alignment')
        u, values, vt=np.linalg.svd(bc.T@ac/len(a));sign=np.ones(3);sign[-1]=np.sign(np.linalg.det(u@vt))
        rotation=u@np.diag(sign)@vt
        scale=float(np.sum(values*sign)/np.mean(np.sum(ac*ac,axis=1)))
        if not np.isfinite(scale) or not 1e-4<scale<1e4: raise ValueError('unstable alignment scale')
        translation=b.mean(0)-scale*rotation@a.mean(0)
        return scale,rotation,translation
    def transformed(values,model):
        scale,rotation,translation=model;return scale*(values@rotation.T)+translation
    rng=np.random.default_rng(int(config.get('geometry_seed',7)))
    order=rng.permutation(len(ids));holdout=order[:max(4,len(ids)//5)];train=order[len(holdout):]
    best=None;best_count=0
    for _ in range(128):
        if check_cancel:check_cancel()
        picked=rng.choice(train,3,replace=False)
        try:model=similarity(source[picked],target[picked])
        except (ValueError,np.linalg.LinAlgError):continue
        mask=np.linalg.norm(transformed(source[train],model)-target[train],axis=1)<=threshold
        if mask.sum()>best_count:best_count=int(mask.sum());best=train[mask]
    if best is None or best_count<12 or best_count<len(train)*.6: return rejected('inconsistent_similarity_alignment')
    try:model=similarity(source[best],target[best])
    except (ValueError,np.linalg.LinAlgError):return rejected('degenerate_similarity_alignment')
    validation_error=np.linalg.norm(transformed(source[holdout],model)-target[holdout],axis=1)
    # Also reject disagreement between the two point maps at a shared track.
    sample_disagreement=[np.max(np.linalg.norm(transformed(np.asarray(list(groups[ids[i]].values())),model)-target[i],axis=1)) for i in np.r_[best,holdout]]
    if np.median(validation_error)>threshold or np.mean(validation_error<=threshold)<.75 or np.median(sample_disagreement)>threshold:
        return rejected('heldout_alignment_or_pointmap_consistency_failed')
    candidates=transformed(np.asarray(learned['points'],np.float64),model)
    sfm_cameras={str(c['image_index']):c for c in sfm['cameras']}
    learned_cameras={str(c['image_index']):c for c in learned['cameras']}
    hulls={}
    for frame,camera in sfm_cameras.items():
        pixel=np.asarray([o[frame] for o in sfm['observations'] if frame in o],np.float32)
        if len(pixel)>=3:hulls[frame]=cv2.convexHull(pixel)
    converted=[];keep=[]
    projected={frame:project_points(candidates,camera) for frame,camera in sfm_cameras.items()}
    nearest=cKDTree(points).query(candidates,k=1)[0]
    for index,track in enumerate(learned['observations']):
        if check_cancel and index%256==0:check_cancel()
        if nearest[index]<extent*.001:continue
        evidence={}
        for frame,pixel in track.items():
            if frame not in sfm_cameras or frame not in learned_cameras or frame not in hulls:continue
            camera=sfm_cameras[frame];old=learned_cameras[frame]
            pixel=np.asarray(pixel,np.float64)*[camera['width']/old['width'],camera['height']/old['height']]
            uv,depth=projected[frame]
            valid=depth[index]>.05 and 0<=uv[index,0]<camera['width'] and 0<=uv[index,1]<camera['height']
            if valid and np.linalg.norm(uv[index]-pixel)<3. and cv2.pointPolygonTest(hulls[frame],tuple(map(float,pixel)),True)>=-2.:
                evidence[frame]=pixel.tolist()
        if len(evidence)>=2:keep.append(index);converted.append(evidence)
    if len(keep)<12:return rejected('insufficient_two_view_supported_completion_points')
    maximum=max(12,min(20000,int(config.get('geometry_completion_max_points',4000))))
    if len(keep)>maximum:
        selected=np.linspace(0,len(keep)-1,maximum,dtype=int);keep=[keep[i] for i in selected];converted=[converted[i] for i in selected]
    result=copy.deepcopy(sfm)
    result['points']=np.concatenate([sfm['points'],candidates[keep].astype(np.float32)])
    result['colors']=np.concatenate([sfm['colors'],np.asarray(learned['colors'])[keep]])
    result['confidence']=np.r_[sfm['confidence'],np.minimum(np.asarray(learned['confidence'])[keep],.49)]
    result['sources']=list(sfm.get('sources',['observed']*len(points)))+['inferred']*len(keep)
    result['observations']=list(sfm['observations'])+converted
    diagnostic.update(status='completed_visible_only',added_points=len(keep),alignment_tracks=len(ids),alignment_train_inliers=best_count,
        alignment_validation_tracks=len(holdout),alignment_validation_median_world_error=float(np.median(validation_error)),
        alignment_validation_threshold_world=threshold,scale=float(model[0]),rotation=model[1].tolist(),translation=model[2].tolist(),
        maximum_reprojection_error_feature_px=3.,support_region='within_SfM_observation_hulls_with_2px_tolerance_in_at_least_two_views')
    return result,diagnostic
