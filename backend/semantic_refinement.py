"""Fixed-geometry, multi-view semantic optimization through an RGB GS renderer.

This is a small, inspectable SAGA/LaGa-inspired baseline, not either paper's
implementation: all cameras share independent per-Gaussian class probabilities;
parts and parents may coexist. No feature encoder, language embedding, generated
view, geometry update, or evaluation annotation is used here. The caller freezes
geometry/opacity and supplies a differentiable *black-background* renderer.

Missing detections are unknown, never negative labels. A class without a positive
observed training mask is left exactly as supplied. Class/instance identity and
hierarchy edges must be established by the caller; names do not create evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from numbers import Integral
import random
import time
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class SemanticRefinementConfig:
    """Predeclared ablations; use held-out GT only after choosing these settings."""

    variant: str = "improved"
    steps: int = 2400
    learning_rate: float = 0.05
    seed: int = 0
    channels_per_step: int = 3
    # Keep frozen experiments on their original class-balanced random sampler.
    # Production may opt into a full frame/class sweep with a guaranteed budget.
    sampling_schedule: str = "legacy"
    minimum_sweeps: int = 0
    checkpoint_steps: tuple[int, ...] = (0, 400, 1200, 2400)
    log_every: int = 100
    initialization_epsilon: float = 0.01
    render_epsilon: float = 1e-6
    balanced_bce: bool = True
    dice_weight: float = 0.25
    boundary_radius: int = 2
    boundary_weight: float = 0.25
    confidence_weighting: bool = True
    confidence_normalization: str = "absolute"
    hierarchy_weight: float = 0.05
    prior_weight: float = 0.002
    prior_confidence_threshold: float = 0.8
    seed_support_threshold: float = 0.5
    regularization_samples: int = 8192
    gradient_clip_norm: float = 5.0

    @classmethod
    def preset(cls, variant: str = "improved", **overrides) -> "SemanticRefinementConfig":
        if variant not in {"simple", "improved"}:
            raise ValueError("variant must be 'simple' or 'improved'")
        values = {"variant": variant}
        if variant == "simple":
            values.update(balanced_bce=False, dice_weight=0.0, boundary_radius=0,
                          boundary_weight=1.0, confidence_weighting=False,
                          hierarchy_weight=0.0, prior_weight=0.0)
        values.update(overrides)
        return cls(**values)

    def validate(self):
        if self.variant not in {"simple", "improved"}:
            raise ValueError("invalid variant")
        for name in ("steps", "seed", "boundary_radius", "minimum_sweeps"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.confidence_normalization not in {"absolute", "relative"}:
            raise ValueError("confidence_normalization must be absolute or relative")
        if self.channels_per_step not in (1, 2, 3):
            raise ValueError("RGB renderer supports channels_per_step in [1, 3]")
        if self.sampling_schedule not in {"legacy", "view_cycle"}:
            raise ValueError("sampling_schedule must be 'legacy' or 'view_cycle'")
        if self.minimum_sweeps and self.sampling_schedule != "view_cycle":
            raise ValueError("minimum_sweeps requires the view_cycle schedule")
        if self.log_every < 1 or self.regularization_samples < 1:
            raise ValueError("log_every and regularization_samples must be positive")
        for name in ("learning_rate", "dice_weight", "hierarchy_weight", "prior_weight",
                     "gradient_clip_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name == "learning_rate" and value == 0):
                raise ValueError(f"invalid {name}")
        for name in ("boundary_weight", "prior_confidence_threshold", "seed_support_threshold"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0 < self.initialization_epsilon < 0.5 or not 0 < self.render_epsilon < 0.5:
            raise ValueError("probability epsilons must be in (0, .5)")
        if any(not isinstance(step, int) or step < 0 for step in self.checkpoint_steps):
            raise ValueError("checkpoint steps must be nonnegative integers")


def boundary_uncertainty_weights(targets: Tensor, radius: int = 2,
                                 boundary_weight: float = 0.25) -> Tensor:
    """Weight both sides of a contour without eroding supervision or image borders.

    targets has shape [C,H,W]. Replicate padding prevents an image edge from
    becoming an artificial object contour, including for 1-pixel dimensions.
    A soft target is thresholded only to locate contours, never to change GT.
    """
    if targets.ndim != 3 or min(targets.shape) < 1:
        raise ValueError("targets must have nonempty shape [C,H,W]")
    if not isinstance(radius, int) or radius < 0 or not 0 <= boundary_weight <= 1:
        raise ValueError("invalid boundary radius or weight")
    weights = torch.ones_like(targets, dtype=torch.float32)
    if radius == 0 or boundary_weight == 1:
        return weights
    binary = (targets >= 0.5).to(torch.float32).unsqueeze(1)
    padded = F.pad(binary, (radius, radius, radius, radius), mode="replicate")
    width = radius * 2 + 1
    dilation = F.max_pool2d(padded, width, stride=1)
    erosion = -F.max_pool2d(-padded, width, stride=1)
    contour = (dilation - erosion).squeeze(1) > 0
    return torch.where(contour, weights * boundary_weight, weights)


def _as_config(config):
    if config is None:
        config = SemanticRefinementConfig.preset()
    elif isinstance(config, Mapping):
        values = dict(config)
        config = SemanticRefinementConfig.preset(values.pop("variant", "improved"), **values)
    if not isinstance(config, SemanticRefinementConfig):
        raise TypeError("config must be SemanticRefinementConfig or a mapping")
    config.validate()
    return config


def _cpu_tensor(value, *, dtype=torch.float32) -> Tensor:
    return torch.as_tensor(value).detach().to(device="cpu", dtype=dtype).contiguous()


def _tree_cpu(value):
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _tree_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_cpu(item) for item in value)
    return value


def _observations(observations, classes, config):
    result = []
    digest = hashlib.sha256()
    seen_ids = set()
    for raw in observations:
        frame_id = str(raw["frame_id"])
        if frame_id in seen_ids:
            raise ValueError("frame_id must be unique; merge same-view masks before refinement")
        seen_ids.add(frame_id)
        compact = "class_indices" in raw
        class_indices = list(raw["class_indices"]) if compact else list(range(classes))
        if any(not isinstance(c, Integral) or isinstance(c, bool) or not 0 <= c < classes
               for c in class_indices) or len(set(class_indices)) != len(class_indices):
            raise ValueError(f"{frame_id}: class_indices must be unique global class indices")
        class_indices = [int(c) for c in class_indices]
        local_classes = len(class_indices)
        targets = _cpu_tensor(raw["targets"])
        if targets.ndim != 3 or targets.shape[0] != local_classes or min(targets.shape[1:]) < 1:
            raise ValueError(f"{frame_id}: targets must have shape [K,H,W] matching class_indices (or C without it)")
        if not torch.isfinite(targets).all() or ((targets < 0) | (targets > 1)).any():
            raise ValueError(f"{frame_id}: targets must be finite probabilities")
        observed = _cpu_tensor(raw["observed"], dtype=torch.bool)
        confidence = _cpu_tensor(raw.get("confidence", torch.ones(local_classes)))
        if observed.shape != (local_classes,) or confidence.shape != (local_classes,):
            raise ValueError(f"{frame_id}: observed/confidence must have shape [K] matching targets")
        if not torch.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
            raise ValueError(f"{frame_id}: confidence must be in [0, 1]")
        valid = _cpu_tensor(raw.get("valid_pixels", torch.ones(targets.shape[1:])))
        if valid.shape != targets.shape[1:] or not torch.isfinite(valid).all() or ((valid < 0) | (valid > 1)).any():
            raise ValueError(f"{frame_id}: valid_pixels must have shape [H,W] and range [0,1]")
        usable = observed & (confidence > 0) & bool(valid.sum() > 0)
        positive = usable & (((targets >= 0.5) & (valid > 0)).flatten(1).any(1))
        weights = (boundary_uncertainty_weights(targets, config.boundary_radius,
                                                config.boundary_weight) * valid.unsqueeze(0)
                   if local_classes else torch.empty_like(targets))
        global_usable = torch.zeros(classes, dtype=torch.bool)
        global_positive = torch.zeros(classes, dtype=torch.bool)
        global_usable[class_indices] = usable
        global_positive[class_indices] = positive
        result.append({"frame_id": frame_id, "camera": raw["camera"], "targets": targets,
                       "observed": global_usable, "positive": global_positive, "confidence": confidence,
                       "weights": weights, "valid_pixels": valid,
                       "class_lookup": {c: local for local, c in enumerate(class_indices)}})
        digest.update(frame_id.encode("utf-8"))
        if compact:
            # Preserve old full-array digests for frozen experiment resumes.
            digest.update(b"compact-class-indices-v1:")
            digest.update(str(class_indices).encode("ascii"))
        # Resume checks use supervision bytes; camera object compatibility remains
        # the caller's responsibility (e.g. a fixed model/camera provenance hash).
        for value in (targets, observed, confidence, valid):
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.numpy().tobytes())
    return result, digest.hexdigest()


def _segmentation_loss(prediction, targets, weights, confidence, config):
    p = prediction.clamp(config.render_epsilon, 1 - config.render_epsilon)
    bce = F.binary_cross_entropy(p, targets, reduction="none")
    if config.balanced_bce:
        positive = (targets >= 0.5).to(weights.dtype)
        w_pos, w_neg = weights * positive, weights * (1 - positive)
        mass_pos, mass_neg = w_pos.sum((1, 2)), w_neg.sum((1, 2))
        loss_pos = (bce * w_pos).sum((1, 2)) / mass_pos.clamp_min(1e-8)
        loss_neg = (bce * w_neg).sum((1, 2)) / mass_neg.clamp_min(1e-8)
        has_pos, has_neg = (mass_pos > 0).float(), (mass_neg > 0).float()
        per_class_bce = (loss_pos * has_pos + loss_neg * has_neg) / (has_pos + has_neg).clamp_min(1)
    else:
        per_class_bce = (bce * weights).sum((1, 2)) / weights.sum((1, 2)).clamp_min(1e-8)
    numerator = 2 * (p * targets * weights).sum((1, 2)) + 1
    denominator = ((p + targets) * weights).sum((1, 2)) + 1
    per_class_dice = 1 - numerator / denominator
    class_weights = confidence if config.confidence_weighting else torch.ones_like(confidence)
    # A single-channel or uniformly unreliable view must still be discounted.
    # Relative normalization is available only to reproduce older experiments.
    denominator = (class_weights.sum().clamp_min(1e-8)
                   if config.confidence_normalization == "relative" else max(1, class_weights.numel()))
    class_weights = class_weights / denominator
    bce_loss = (per_class_bce * class_weights).sum()
    dice_loss = (per_class_dice * class_weights).sum()
    return bce_loss + config.dice_weight * dice_loss, bce_loss, dice_loss


def _view_cycle(eligible_classes, channels_per_step, seed, sweep):
    """One deterministic visit per usable frame/class pair, in RGB-sized groups.

    Frames are interleaved, so a frame with many labels cannot delay all other
    frames until its labels have been exhausted. Empty/unknown pairs are absent;
    explicit negative observations of a globally active class remain eligible.
    The seed and sweep number reconstruct this schedule exactly on resume.
    """
    rng = random.Random(f"semantic-view-cycle-v1:{seed}:{sweep}")
    groups = {}
    for view_index, classes in enumerate(eligible_classes):
        shuffled = list(classes)
        rng.shuffle(shuffled)
        if shuffled:
            groups[view_index] = [shuffled[start:start + channels_per_step]
                                  for start in range(0, len(shuffled), channels_per_step)]
    result = []
    for group_index in range(max((len(value) for value in groups.values()), default=0)):
        frames = [view_index for view_index, value in groups.items() if group_index < len(value)]
        rng.shuffle(frames)
        result.extend((view_index, groups[view_index][group_index]) for view_index in frames)
    return result


def _stage_observation(observation, selected, device):
    """Only the current <=3 supervision channels are copied onto the device.

    _observations keeps the complete mask collection on CPU. This avoids GPU
    mask memory growing with the total number of training cameras or classes.
    """
    local_selected = [observation["class_lookup"][c] for c in selected]
    return tuple(observation[key][local_selected].to(device)
                 for key in ("targets", "weights", "confidence"))


def _hierarchy_support(value, declared_edges, active_edges, count, classes):
    """Validate optional per-edge global point IDs without inventing support."""
    if value is None:
        return None, None
    if not isinstance(value, Mapping):
        raise ValueError('hierarchy_support_indices must map (parent, child) to unique point IDs')
    result, rows = {}, []
    digest = hashlib.sha256(str(('geometry_hierarchy_support_v1', count, classes)).encode())
    declared = {tuple(edge) for edge in declared_edges}
    for edge in value:
        if (not isinstance(edge, tuple) or len(edge) != 2 or
                any(type(index) is not int for index in edge) or edge not in declared):
            raise ValueError('Hierarchy support references an undeclared parent-child edge')
    for edge, indices in sorted(value.items()):
        raw = torch.as_tensor(indices).detach().cpu()
        if raw.ndim != 1 or (raw.numel() and raw.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)):
            raise ValueError('Hierarchy support IDs must be a one-dimensional integer sequence')
        ids = raw.to(torch.int64).sort().values.contiguous()
        if ids.numel() and ((ids < 0).any() or (ids >= count).any() or len(torch.unique(ids)) != len(ids)):
            raise ValueError('Hierarchy support IDs must be unique and within the full Gaussian count')
        if edge not in active_edges:
            continue
        checksum = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
        digest.update(str((edge, len(ids), checksum)).encode())
        rows.append({'parent': edge[0], 'child': edge[1], 'point_count': len(ids), 'sha256': checksum})
        result[edge] = ids
    return result, {'format': 'geometry_hierarchy_support_v1', 'gaussian_count': count,
                    'class_count': classes, 'edges': rows, 'sha256': digest.hexdigest()}


def refine_semantics(
    initial_probs,
    observations: Sequence[Mapping[str, Any]],
    render_probability_fn: Callable[[Any, Tensor], Tensor],
    hierarchy_edges: Sequence[tuple[int, int]] = (),
    config: SemanticRefinementConfig | Mapping[str, Any] | None = None,
    checkpoint_callback: Callable[[int, Tensor, dict], None] | None = None,
    *,
    resume_state: Mapping[str, Any] | None = None,
    state_callback: Callable[[int, dict], None] | None = None,
    progress_callback: Callable[[int, dict], None] | None = None,
    hierarchy_support_indices: Mapping[tuple[int, int], Sequence[int]] | None = None,
) -> tuple[Tensor, dict]:
    """Optimize semantics and return (CPU probabilities[N,C], JSON-safe stats).

    Each observation contains frame_id, camera, targets[C,H,W], observed[C], and
    confidence[C] in [0,1]; valid_pixels[H,W] is optional. For compact CPU mask
    storage, supply class_indices[K] with global class indices and use targets,
    observed, confidence with K channels. Omitted and false-observed pairs are
    unknown, including their apparent background; they receive no data loss.

    renderer(camera, colors[N,3]) must return differentiable [3,H,W] alpha-
    composited values with zero background. It must preserve input order and use
    fixed opacity/geometry. Three independent class fields share one RGB call;
    unused channels are zero and never supervised. This creates cross-view
    consistency through a single shared 3D field, without identity association
    between objects with the same name or an extra pairwise contrastive loss.

    checkpoint_callback(step, probabilities_cpu, metadata) runs at configured
    steps and the final step. step=0 reflects the documented probability clamp.
    state_callback(step, state) additionally receives logits, Adam, sampling RNG,
    schedule cursor, and visit counts for exact CPU continuation. A probabilities-only checkpoint
    is a warm start, not an exact resume. On CUDA, kernel determinism remains the
    caller's responsibility. Pass the *same initial_probs* on exact resume.
    progress_callback(step, row) receives each logged training loss and elapsed
    optimizer time (step 1, log_every, and checkpoints); callback time is excluded.
    view_cycle visits every usable observed frame/active-class pair once per
    sweep. minimum_sweeps may extend steps; planned_steps reports that budget.
    hierarchy_support_indices optionally supplies verified global point IDs per
    parent-child edge. It enables hierarchy penalties without a semantic prior;
    prior anchoring still uses confident positive seeds only. None preserves the
    historical loss, RNG sequence and resume format exactly.
    """
    call_started = time.perf_counter()
    config = _as_config(config)
    initial = torch.as_tensor(initial_probs).detach().to(dtype=torch.float32)
    if initial.ndim != 2 or min(initial.shape) < 1:
        raise ValueError("initial_probs must have nonempty shape [N,C]")
    if not torch.isfinite(initial).all() or ((initial < 0) | (initial > 1)).any():
        raise ValueError("initial_probs must contain finite probabilities")
    device = initial.device
    count, classes = initial.shape
    prepared, observation_digest = _observations(observations, classes, config)
    active = sorted({c for obs in prepared for c in torch.where(obs["positive"])[0].tolist()})
    active_set = set(active)
    mapping = {c: index for index, c in enumerate(active)}
    eligible_classes = [[c for c in active if obs["observed"][c]] for obs in prepared]
    sweep_steps = sum(math.ceil(len(eligible) / config.channels_per_step) for eligible in eligible_classes)
    effective_steps = max(config.steps, sweep_steps * config.minimum_sweeps)
    edges = []
    for edge in hierarchy_edges:
        if len(edge) != 2:
            raise ValueError("hierarchy edges must be (parent, child) pairs")
        parent, child = edge
        if not isinstance(parent, int) or not isinstance(child, int) or not 0 <= parent < classes or not 0 <= child < classes or parent == child:
            raise ValueError("hierarchy edge has invalid class indices")
        if parent in active_set and child in active_set and (parent, child) not in edges:
            edges.append((parent, child))
    # Reject cycles: a taxonomy is a DAG, although a part can have multiple parents.
    adjacency = {c: [] for c in range(classes)}
    for parent, child in hierarchy_edges:
        adjacency[parent].append(child)
    visiting, visited = set(), set()

    def visit(node):
        if node in visiting:
            raise ValueError("hierarchy edges must not contain cycles")
        if node in visited:
            return
        visiting.add(node)
        for child in adjacency[node]:
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for c in range(classes):
        visit(c)
    geometry_support, support_identity = _hierarchy_support(
        hierarchy_support_indices, hierarchy_edges, edges, count, classes)
    geometry_support = None if geometry_support is None else {
        edge: ids.to(device=device) for edge, ids in geometry_support.items()}
    rng = random.Random(config.seed)
    torch_rng = torch.Generator(device="cpu").manual_seed(config.seed)
    view_class_visits = [[0] * classes for _ in prepared]
    class_visits = [0] * classes
    view_visits = [0] * len(prepared)
    anchor_queue = []
    current_sweep = -1
    cycle = []
    history = []
    class_views = {c: [i for i, obs in enumerate(prepared) if obs["positive"][c]] for c in active}
    initial_cpu = initial.cpu().contiguous()
    initial_digest = hashlib.sha256(initial_cpu.numpy().tobytes()).hexdigest()
    callback_seconds = 0.0
    elapsed_before_resume = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    init_selected = initial[:, active].clone()
    epsilon = config.initialization_epsilon
    logits = torch.nn.Parameter(torch.logit(init_selected.clamp(epsilon, 1 - epsilon)))
    optimizer = torch.optim.Adam([logits], lr=config.learning_rate) if active else None
    support = torch.where(init_selected.max(1).values >= config.seed_support_threshold)[0] if active else torch.empty(0, dtype=torch.long, device=device)
    local_edges = [(mapping[parent], mapping[child]) for parent, child in edges]
    start_step = 0
    config_dict = asdict(config)
    # Budget/checkpoint changes may extend a run, never alter its sampler or loss.
    compatibility_config = {key: value for key, value in config_dict.items()
                            if key not in {"steps", "minimum_sweeps", "checkpoint_steps", "log_every"}}
    if resume_state is not None:
        previous_config = dict(resume_state.get("compatibility_config", {}))
        previous_config.setdefault("sampling_schedule", "legacy")
        previous_config.pop("minimum_sweeps", None)
        if (resume_state.get("format_version") != 1 or tuple(resume_state.get("shape", ())) != tuple(initial.shape)
                or resume_state.get("active_classes") != active
                or resume_state.get("observation_digest") != observation_digest
                or resume_state.get("initial_digest") != initial_digest
                or previous_config != compatibility_config
                or resume_state.get("hierarchy_edges") != edges
                or resume_state.get('hierarchy_support_identity') != support_identity):
            raise ValueError("resume state is incompatible with shape, evidence, initialization, hierarchy, or loss configuration")
        start_step = int(resume_state["step"])
        if start_step < 0 or start_step > effective_steps:
            raise ValueError("resume step is outside requested schedule")
        with torch.no_grad():
            logits.copy_(resume_state["logits"].to(device=device, dtype=logits.dtype))
        if not torch.isfinite(logits).all():
            raise ValueError("resume logits are non-finite")
        if optimizer:
            optimizer.load_state_dict(resume_state["optimizer"])
        rng.setstate(resume_state["python_rng_state"])
        torch_rng.set_state(resume_state["torch_rng_state"].cpu())
        anchor_queue = list(resume_state["anchor_queue"])
        class_visits = list(resume_state["class_visits"])
        view_visits = list(resume_state["view_visits"])
        view_class_visits = [list(row) for row in resume_state["view_class_visits"]]
        history = list(resume_state["history"])
        elapsed_before_resume = float(resume_state.get("training_seconds", 0))

    # Exclude validation, initialization, checkpoint loading, and work submitted
    # by a caller before this optimizer. Synchronize before starting the clock.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    setup_seconds = time.perf_counter() - call_started
    start_time = time.perf_counter()

    def probabilities():
        result = initial_cpu.clone()
        if active:
            result[:, active] = logits.detach().sigmoid().cpu()
        return result

    def training_seconds():
        return elapsed_before_resume + time.perf_counter() - start_time - callback_seconds

    def coverage(step):
        eligible_pairs = [(i, c) for i, classes_for_view in enumerate(eligible_classes) for c in classes_for_view]
        uncovered = [{"frame_id": prepared[i]["frame_id"], "class_index": c}
                     for i, c in eligible_pairs if not view_class_visits[i][c]]
        return {"sampling_schedule": config.sampling_schedule,
                "sweep_steps": sweep_steps, "min_required_steps": sweep_steps * config.minimum_sweeps,
                "minimum_sweeps": config.minimum_sweeps,
                "planned_steps": effective_steps,
                "planned_sweeps": effective_steps // sweep_steps if sweep_steps else 0,
                "completed_sweeps": (step // sweep_steps if sweep_steps else 0)
                    if config.sampling_schedule == "view_cycle" else None,
                "minimum_pair_visits": min((view_class_visits[i][c] for i, c in eligible_pairs), default=0),
                "observed_frame_class_pairs": len(eligible_pairs),
                "visited_frame_class_pairs": len(eligible_pairs) - len(uncovered),
                "uncovered_observed_frame_class_pairs": uncovered,
                "eligible_frames": sum(bool(value) for value in eligible_classes),
                "uncovered_observed_frames": [prepared[i]["frame_id"] for i, value in enumerate(eligible_classes)
                                               if value and not view_visits[i]]}

    def checkpoint(step):
        nonlocal callback_seconds
        if not checkpoint_callback and not state_callback:
            return
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        now = time.perf_counter()
        metadata = {"format_version": 1, "step": step, "shape": [count, classes],
                    "active_classes": active, "unknown_classes": sorted(set(range(classes)) - active_set),
                    "variant": config.variant, "training_seconds": training_seconds(),
                    "planned_steps": effective_steps, "coverage": coverage(step),
                    "observation_digest": observation_digest, "initial_digest": initial_digest,
                    "initialization_epsilon": epsilon, "config": config_dict}
        if support_identity is not None:
            metadata['hierarchy_support_identity'] = support_identity
        if checkpoint_callback:
            checkpoint_callback(step, probabilities(), metadata)
        if state_callback:
            state_callback(step, {**metadata, "compatibility_config": compatibility_config,
                                  "hierarchy_edges": edges, "logits": logits.detach().cpu().clone(),
                                  "optimizer": _tree_cpu(optimizer.state_dict()) if optimizer else None,
                                  "python_rng_state": rng.getstate(), "torch_rng_state": torch_rng.get_state().clone(),
                                  "anchor_queue": list(anchor_queue), "class_visits": list(class_visits),
                                  "schedule_state": {"sampling_schedule": config.sampling_schedule,
                                                     "sweep_steps": sweep_steps,
                                                     "sweep_index": step // sweep_steps if sweep_steps else 0,
                                                     "step_in_sweep": step % sweep_steps if sweep_steps else 0},
                                  "view_visits": list(view_visits), "view_class_visits": [list(row) for row in view_class_visits],
                                  "history": list(history)})
        callback_seconds += time.perf_counter() - now

    checkpoints = set(config.checkpoint_steps) | {effective_steps}
    if start_step in checkpoints:
        checkpoint(start_step)
    completed_step = start_step
    for step in range(start_step + 1, effective_steps + 1) if active else ():
        if config.sampling_schedule == "view_cycle":
            sweep = (step - 1) // sweep_steps
            if sweep != current_sweep:
                cycle = _view_cycle(eligible_classes, config.channels_per_step, config.seed, sweep)
                current_sweep = sweep
            view_index, selected = cycle[(step - 1) % sweep_steps]
        else:
            if not anchor_queue:
                anchor_queue = list(active)
                rng.shuffle(anchor_queue)
            anchor = anchor_queue.pop()
            candidates = class_views[anchor]
            minimum = min(view_class_visits[i][anchor] for i in candidates)
            view_index = rng.choice([i for i in candidates if view_class_visits[i][anchor] == minimum])
            extras = [c for c in active if c != anchor and prepared[view_index]["observed"][c]]
            rng.shuffle(extras)
            extras.sort(key=lambda c: (view_class_visits[view_index][c], class_visits[c]))
            selected = [anchor] + extras[:config.channels_per_step - 1]
        obs = prepared[view_index]
        local_selected = [mapping[c] for c in selected]
        colors = logits[:, local_selected].sigmoid()
        if len(selected) < 3:
            colors = F.pad(colors, (0, 3 - len(selected)), value=0)
        prediction = render_probability_fn(obs["camera"], colors.contiguous())
        expected_shape = (3, *obs["targets"].shape[1:])
        if not isinstance(prediction, Tensor) or tuple(prediction.shape) != expected_shape:
            raise ValueError(f"renderer must return {expected_shape} for frame {obs['frame_id']}")
        if prediction.device != device or not prediction.requires_grad:
            raise ValueError("renderer must return a differentiable tensor on the logits device")
        if not torch.isfinite(prediction).all():
            raise FloatingPointError(f"non-finite render at step {step}, frame {obs['frame_id']}")
        if bool((prediction.detach() < -1e-4).any()) or bool((prediction.detach() > 1.0001).any()):
            raise ValueError("renderer returned values outside [0,1]; use black-background probability rendering")
        targets, weights, confidence = _stage_observation(obs, selected, device)
        data_loss, bce, dice = _segmentation_loss(prediction[:len(selected)], targets, weights, confidence, config)
        hierarchy_loss = logits.new_zeros(())
        prior_loss = logits.new_zeros(())
        if len(support) and ((local_edges and config.hierarchy_weight and geometry_support is None) or config.prior_weight):
            sample_count = min(config.regularization_samples, len(support))
            # Sampling with replacement is O(sample_count), not O(N) per step.
            sampled = support[torch.randint(len(support), (sample_count,), generator=torch_rng).to(device)]
            sample_probs = logits[sampled].sigmoid()
            if local_edges and config.hierarchy_weight and geometry_support is None:
                parent_idx, child_idx = zip(*local_edges)
                hierarchy_loss = F.relu(sample_probs[:, child_idx] - sample_probs[:, parent_idx]).mean()
            if config.prior_weight:
                prior = init_selected[sampled]
                # Low initial probabilities often mean "unseen", not background.
                # Anchor only confident positive seeds and permit new membership.
                reliable = prior >= config.prior_confidence_threshold
                if reliable.any():
                    prior_loss = ((sample_probs - prior).square() * reliable).sum() / reliable.sum()
        if geometry_support is not None and config.hierarchy_weight:
            terms = []
            for (parent, child), ids in geometry_support.items():
                if not len(ids):
                    continue
                sampled = ids[torch.randint(len(ids), (min(config.regularization_samples, len(ids)),),
                                            generator=torch_rng).to(device)]
                pair = logits[sampled][:, [mapping[parent], mapping[child]]].sigmoid()
                terms.append(F.relu(pair[:, 1] - pair[:, 0]).mean())
            if terms:
                hierarchy_loss = torch.stack(terms).mean()
        loss = data_loss + config.hierarchy_weight * hierarchy_loss + config.prior_weight * prior_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite semantic loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if logits.grad is None or not torch.isfinite(logits.grad).all():
            raise FloatingPointError(f"missing or non-finite semantic gradient at step {step}")
        if config.gradient_clip_norm:
            torch.nn.utils.clip_grad_norm_([logits], config.gradient_clip_norm, error_if_nonfinite=True)
        optimizer.step()
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"non-finite logits at step {step}")
        view_visits[view_index] += 1
        for c in selected:
            class_visits[c] += 1
            view_class_visits[view_index][c] += 1
        completed_step = step
        if step == 1 or step % config.log_every == 0 or step in checkpoints:
            row = {"step": step, "frame_id": obs["frame_id"], "classes": selected,
                   "loss": float(loss.detach()), "bce": float(bce.detach()),
                   "dice": float(dice.detach()), "hierarchy_loss": float(hierarchy_loss.detach()),
                   "prior_loss": float(prior_loss.detach())}
            history.append(row)
            if progress_callback:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                callback_start = time.perf_counter()
                progress_callback(step, {**row, "training_seconds": training_seconds(),
                                         "planned_steps": effective_steps,
                                         "coverage": coverage(step)})
                callback_seconds += time.perf_counter() - callback_start
        if step in checkpoints:
            checkpoint(step)
    if not active and completed_step not in checkpoints:
        checkpoint(completed_step)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = training_seconds()
    result = probabilities()
    stats = {"format_version": 1, "status": "completed" if active else "no_positive_observations",
             "variant": config.variant, "config": config_dict, "requested_steps": config.steps,
             "planned_steps": effective_steps, "coverage": coverage(completed_step),
             "completed_steps": completed_step, "resumed_from_step": start_step,
             "shape": [count, classes], "device": str(device), "seed": config.seed,
             "active_classes": active, "unknown_classes": sorted(set(range(classes)) - active_set),
             "regularization_support_points": len(support), "hierarchy_edges_used": edges,
             "hierarchy_edges_skipped_unobserved": sum(parent not in active_set or child not in active_set
                                                        for parent, child in hierarchy_edges),
             "observation_digest": observation_digest, "initial_digest": initial_digest,
             "setup_seconds": setup_seconds, "training_seconds": elapsed,
             "callback_seconds": callback_seconds,
             "timing_scope": "optimizer loop, including rendering and logging reductions; excludes setup and callbacks",
             "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
             "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
             "observation_storage_device": "cpu",
             "mask_transfer_policy": "selected_channels_per_step",
             "cpu_observation_tensor_bytes": sum(value.numel() * value.element_size()
                                                  for obs in prepared for value in obs.values() if isinstance(value, Tensor)),
             "class_visits": class_visits,
             "view_visits": {obs["frame_id"]: view_visits[i] for i, obs in enumerate(prepared)},
             "view_class_visits": {obs["frame_id"]: view_class_visits[i] for i, obs in enumerate(prepared)},
             "positive_views_per_class": [len(class_views.get(c, ())) for c in range(classes)],
             "history": history,
             "consistency_mechanism": "shared_per_gaussian_multilabel_field_rendered_from_real_training_views",
             "limitations": ["fixed geometry and caller-supplied masks/identity/hierarchy",
                             "no independent 3D ground truth or identity tracking inside optimizer",
                             "hierarchy loss is a regularizer, not a semantic accuracy metric",
                             "unobserved classes preserve the supplied prior; unknown is not a negative label",
                             "CUDA renderer may use nondeterministic atomic operations"]}
    if support_identity is not None:
        stats['hierarchy_support_identity'] = support_identity
        stats['hierarchy_geometry_support_points_per_edge'] = {
            f'{parent}:{child}': len(ids) for (parent, child), ids in geometry_support.items()}
        stats['hierarchy_support_source'] = 'explicit_per_edge_visible_geometry'
        stats['prior_support_source'] = 'confident_initial_semantic_seeds_only'
    return result, stats
