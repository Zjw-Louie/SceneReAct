from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from src.eval import create_floor_plan_polygon, compute_oob, eval_scene, get_xz_bbox_from_obj
from src.respace import ReSpace
from src.viz import render_annotated_top_view
from scorer.gpt_vl_image_describe_v10_role_refactor import (
    GPTVLMovePromptGeneratorV5,
    ObjectEdit,
    apply_edits_to_scene,
    parse_move_prompt,
    quaternion_from_yaw,
    yaw_from_quaternion,
)
from scorer.scene_role_layout import (
    REL_TYPES,
    RoleGraph,
    build_category_map,
    build_role_based_relation_priors,
    choose_open_side_for_anchor,
    compute_functional_loss,
    distance_to_nearest_wall_xz,
    infer_role_graph,
    major_anchor_indices,
    obj_diag_size_xz,
    obj_size_xz,
    optimization_stage_order,
    post_refine_role_layout,
    room_center_xz,
    target_pose_for_attachment,
    xz_dist,
)


# ------------------------------
# I/O and timing
# ------------------------------

def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _now() -> float:
    return time.perf_counter()


def _log(message: str) -> None:
    print(message, flush=True)


def _scene_state_hash(scene: Dict[str, Any]) -> str:
    payload = json.dumps(scene, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _compose_step_extra_context(
    base_context: str,
    history_records: Sequence[Dict[str, Any]],
    planner_mode: str,
    max_history_steps: int,
) -> str:
    parts: List[str] = []
    if base_context.strip():
        parts.append(base_context.strip())
    parts.append(
        "HISTORY-AWARE ITERATIVE PLANNING MODE:\n"
        "Use the step history below as additional context. Avoid repeating rejected edits. "
        "Diagnose why the previous proposal failed, then propose a different fix-as-you-go action plan. "
        "Prefer preserving global functional zones while repairing local relation issues."
    )
    parts.append(f"CURRENT PLANNER MODE: {planner_mode}")
    if history_records:
        parts.append("RECENT STEP HISTORY:")
        for rec in list(history_records)[-max_history_steps:]:
            parts.append(
                f"- step {rec.get('step', -1):02d} mode={rec.get('planner_mode', 'unknown')} "
                f"accepted={rec.get('accepted', False)} reject={rec.get('reject_reason', 'none')} "
                f"score {rec.get('score_before', 0.0):.4f}->{rec.get('score_after', 0.0):.4f}; "
                f"rel {rec.get('rel_before', 0.0):.4f}->{rec.get('rel_after', 0.0):.4f}; "
                f"func {rec.get('func_before', 0.0):.4f}->{rec.get('func_after', 0.0):.4f}; "
                f"struct {rec.get('struct_before', 0.0):.4f}->{rec.get('struct_after', 0.0):.4f}; "
                f"zones {rec.get('zone_before', 0)}->{rec.get('zone_after', 0)}; "
                f"mono {rec.get('mono_before', 0.0):.3f}->{rec.get('mono_after', 0.0):.3f}"
            )
            if rec.get('diagnosis'):
                parts.append(f"  diagnosis: {rec.get('diagnosis')}")
        repeated_relation = [r for r in history_records if (not r.get('accepted')) and str(r.get('reject_reason', '')).startswith('relation_worse')]
        if repeated_relation:
            parts.append(
                "HISTORY GUIDANCE: Previous proposals often improved structure or opened new zones but were rejected because relation loss got worse. "
                "Do not blindly repeat the same macro relayout. Either (a) keep the new secondary zone but add supporting relation-consistent bridge edits, or (b) propose a smaller relayout that preserves the dominant-anchor semantics while reducing monopoly."
            )
        repeated_same = [r for r in history_records if str(r.get('reject_reason', '')).startswith('repeat_')]
        if repeated_same:
            parts.append(
                "ANTI-LOOP GUIDANCE: A near-duplicate candidate has already been rejected. You must diversify the next plan: change target objects, change target anchor, or change the repair order."
            )
    return "\n\n".join(parts).strip()


def _history_relayout_worthy(
    score_before: float,
    score_after: float,
    struct_before: Dict[str, Any],
    struct_after: Dict[str, Any],
    cfg: Config,
) -> bool:
    score_gain = float(score_before) - float(score_after)
    structure_gain = float(struct_before.get('structure_loss', 0.0)) - float(struct_after.get('structure_loss', 0.0))
    monopoly_gain = float(struct_before.get('max_zone_ratio', 0.0)) - float(struct_after.get('max_zone_ratio', 0.0))
    zone_gain = int(float(struct_after.get('zone_count', 0.0))) - int(float(struct_before.get('zone_count', 0.0)))
    return (
        score_gain >= cfg.relayout_accept_min_score_gain
        and structure_gain >= cfg.relayout_accept_min_structure_gain
        and monopoly_gain >= cfg.relayout_accept_min_monopoly_gain
        and zone_gain >= cfg.relayout_accept_min_zone_gain
    )


def _rescore_with_counterfactual_priors(
    scene: Dict[str, Any],
    role_graph: RoleGraph,
    timing_stats: Optional[TimingStats],
    cfg: Config,
) -> Tuple[float, Dict[str, Any], float, float, List[Dict[str, Any]]]:
    rewritten_priors = build_role_based_relation_priors(scene, role_graph)
    rewritten_score, rewritten_metrics, rewritten_rel, rewritten_func = _score_scene_full(
        scene,
        role_graph,
        rewritten_priors,
        timing_stats,
        cfg,
    )
    return rewritten_score, rewritten_metrics, rewritten_rel, rewritten_func, rewritten_priors


@dataclass
class TimingStats:
    render_sec: float = 0.0
    vlm_sec: float = 0.0
    optimize_sec: float = 0.0
    eval_sec: float = 0.0


@dataclass
class Config:
    max_steps: int = 3
    max_rounds: int = 2
    max_objects_per_round: int = 8
    proxy_topk: int = 3
    move_prompt_temperature: float = 0.2
    move_prompt_max_tokens: int = 1200
    move_prompt_retries: int = 2
    relation_prior_retries: int = 2
    relation_prior_temperature: float = 0.0
    relation_prior_max_tokens: int = 900
    relation_prior_confidence: float = 0.55
    use_vlm_relation_priors: bool = True
    refresh_vlm_relation_priors_every_step: bool = False
    merge_vlm_with_deterministic: bool = True
    vlm_prior_weight_scale: float = 0.35
    freeze_deterministic_priors: bool = False
    local_repair_passes: int = 1
    full_repair_after_refine_passes: int = 1
    stop_when_valid_pbl: bool = True
    cleanup_only_after_valid_pbl: bool = True
    max_steps_after_valid_pbl: int = 1
    max_objects_after_valid_pbl: int = 4
    max_rounds_after_valid_pbl: int = 1
    min_score_improve_after_valid_pbl: float = 0.02
    max_rel_increase_after_valid_pbl: float = 0.12
    max_func_increase_after_valid_pbl: float = 0.12
    max_score_increase_prevalid: float = 0.06
    max_rel_increase_prevalid: float = 0.55
    max_func_increase_prevalid: float = 0.35
    render_final: bool = True
    monotonic_eps: float = 1e-12
    step_xy: float = 0.22
    step_yaw: float = 15.0
    anchor_lock_pbl_threshold: float = 0.08
    role_refine_blend: float = 0.66
    valid_pbl_threshold: float = 0.10
    stop_score_threshold: float = 0.80
    skip_post_refine_when_valid_pbl: bool = True
    candidate_filter_reintroduced_pbl: bool = True
    use_structural_guard: bool = True
    min_open_space_ratio: float = 0.42
    max_zone_monopoly_ratio: float = 0.72
    corridor_width_ratio: float = 0.18
    max_open_space_drop_after_valid_pbl: float = 0.06
    max_monopoly_increase_after_valid_pbl: float = 0.08
    max_spread_increase_after_valid_pbl: float = 0.10
    max_flow_increase_after_valid_pbl: float = 0.10
    require_zone_count_preserve_after_valid_pbl: bool = True
    add_zone_release_candidates: bool = True
    zone_release_inset_ratio: float = 0.16
    enable_zone_layout_candidates: bool = True
    zone_layout_trigger_ratio: float = 0.76
    max_zone_layout_candidates: int = 4
    zone_layout_topk_objects: int = 3
    zone_layout_min_score_improve: float = 0.01
    zone_layout_anchor_inset_ratio: float = 0.14
    zone_layout_secondary_cluster_radius: float = 0.55
    zone_layout_pair_table_chair: bool = True
    use_iteration_history: bool = True
    max_history_steps: int = 4
    repeat_reject_patience: int = 2
    adaptive_relation_tradeoff: bool = True
    rewrite_priors_on_relayout: bool = True
    refresh_vlm_relation_priors_on_stagnation: bool = True
    force_history_replan_on_monopoly: bool = True
    relayout_accept_min_score_gain: float = 0.03
    relayout_accept_min_structure_gain: float = 0.01
    relayout_accept_min_monopoly_gain: float = 0.15
    relayout_accept_min_zone_gain: int = 1


# ------------------------------
# geometry helpers
# ------------------------------

def _normalize_angle(deg: float) -> float:
    return deg % 360.0


def _angle_diff(a: float, b: float) -> float:
    d = abs(_normalize_angle(a) - _normalize_angle(b))
    return min(d, 360.0 - d)


def _forward_vec_from_yaw(yaw_deg: float) -> Tuple[float, float]:
    rad = math.radians(yaw_deg)
    return math.sin(rad), math.cos(rad)


def _signed_proj(ax: float, az: float, bx: float, bz: float, vx: float, vz: float) -> float:
    return (bx - ax) * vx + (bz - az) * vz


def _deepcopy_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(scene)


def _clone_scene_with_updated_object(scene: Dict[str, Any], idx: int, new_obj: Dict[str, Any]) -> Dict[str, Any]:
    sc = dict(scene)
    objs = list(scene.get("objects", []))
    objs[idx] = new_obj
    sc["objects"] = objs
    return sc


def _get_floor_polygon(scene: Dict[str, Any]) -> Optional[ShapelyPolygon]:
    bounds_bottom = scene.get("bounds_bottom")
    if not isinstance(bounds_bottom, list) or len(bounds_bottom) < 3:
        return None
    try:
        return create_floor_plan_polygon(bounds_bottom)
    except Exception:
        return None


def _find_nearest_wall_yaw(scene: Dict[str, Any], pos: Sequence[float]) -> Optional[float]:
    pts = [(float(p[0]), float(p[2])) for p in scene.get("bounds_bottom", []) if isinstance(p, list) and len(p) >= 3]
    if len(pts) < 3:
        return None
    center_x, center_z = room_center_xz(scene)
    best_dist = float("inf")
    best_normal = None
    for i in range(len(pts)):
        ax, az = pts[i]
        bx, bz = pts[(i + 1) % len(pts)]
        abx, abz = bx - ax, bz - az
        apx, apz = pos[0] - ax, pos[2] - az
        denom = abx * abx + abz * abz
        if denom < 1e-12:
            continue
        t = max(0.0, min(1.0, (apx * abx + apz * abz) / denom))
        proj_x = ax + t * abx
        proj_z = az + t * abz
        dist = math.hypot(pos[0] - proj_x, pos[2] - proj_z)
        if dist >= best_dist:
            continue
        nx, nz = -abz, abx
        nl = math.hypot(nx, nz)
        if nl < 1e-9:
            continue
        nx, nz = nx / nl, nz / nl
        if nx * (center_x - proj_x) + nz * (center_z - proj_z) < 0:
            nx, nz = -nx, -nz
        best_dist = dist
        best_normal = (nx, nz)
    if best_normal is None:
        return None
    nx, nz = best_normal
    return _normalize_angle(math.degrees(math.atan2(nx, nz)))


def _nearest_parallel_wall_yaw(scene: Dict[str, Any], pos: Sequence[float], current_yaw: Optional[float] = None) -> Optional[float]:
    wall_yaw = _find_nearest_wall_yaw(scene, pos)
    if wall_yaw is None:
        return None
    options = [_normalize_angle(wall_yaw + 90.0), _normalize_angle(wall_yaw + 270.0)]
    if current_yaw is None:
        return options[0]
    return min(options, key=lambda y: _angle_diff(y, current_yaw))


def _nearest_normal_axis_yaw(scene: Dict[str, Any], pos: Sequence[float], current_yaw: Optional[float] = None) -> Optional[float]:
    wall_yaw = _find_nearest_wall_yaw(scene, pos)
    if wall_yaw is None:
        return None
    options = [_normalize_angle(wall_yaw), _normalize_angle(wall_yaw + 180.0)]
    if current_yaw is None:
        return options[0]
    return min(options, key=lambda y: _angle_diff(y, current_yaw))


def _axis_angle_diff(a: float, b: float) -> float:
    return min(_angle_diff(a, b), _angle_diff(a, _normalize_angle(b + 180.0)))


def _pair_target_dist(a: Dict[str, Any], b: Dict[str, Any], alpha: float = 0.35, bias: float = 0.15) -> float:
    return alpha * (obj_diag_size_xz(a) + obj_diag_size_xz(b)) + bias


# ------------------------------
# local repair
# ------------------------------

def _compute_oob_push_direction(scene: Dict[str, Any], obj: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    floor_polygon = _get_floor_polygon(scene)
    if floor_polygon is None:
        return None
    oob = compute_oob(obj, floor_polygon, scene.get("bounds_bottom", []), scene.get("bounds_top", []), is_debug=False)
    if oob <= 1e-8:
        return None
    pos = obj.get("pos", [0.0, 0.0, 0.0])
    point = ShapelyPoint(pos[0], pos[2])
    if floor_polygon.contains(point):
        centroid = floor_polygon.centroid
        dx, dz = centroid.x - pos[0], centroid.y - pos[2]
    else:
        nearest = floor_polygon.exterior.interpolate(floor_polygon.exterior.project(point))
        dx, dz = nearest.x - pos[0], nearest.y - pos[2]
    norm = math.hypot(dx, dz)
    if norm < 1e-9:
        return None
    return dx / norm, dz / norm, oob


def _collision_neighbors(scene: Dict[str, Any], idx: int) -> List[Tuple[int, float, float, float]]:
    objs = scene.get("objects", [])
    if not (0 <= idx < len(objs)):
        return []
    try:
        bbox_a, _, ya0, ya1 = get_xz_bbox_from_obj(objs[idx])
    except Exception:
        return []
    results = []
    pa = objs[idx].get("pos", [0.0, 0.0, 0.0])
    for j, other in enumerate(objs):
        if j == idx:
            continue
        try:
            bbox_b, _, yb0, yb1 = get_xz_bbox_from_obj(other)
        except Exception:
            continue
        y_overlap = max(0.0, min(ya1, yb1) - max(ya0, yb0))
        if y_overlap <= 0:
            continue
        inter = bbox_a.intersection(bbox_b)
        if inter.is_empty or inter.area < 1e-8:
            continue
        pb = other.get("pos", [0.0, 0.0, 0.0])
        dx, dz = pa[0] - pb[0], pa[2] - pb[2]
        norm = math.hypot(dx, dz)
        if norm < 1e-9:
            dx, dz = 1.0, 0.0
        else:
            dx, dz = dx / norm, dz / norm
        results.append((j, dx, dz, float(inter.area)))
    return results


def _project_inside_room(scene: Dict[str, Any], idx: int) -> None:
    obj = scene.get("objects", [])[idx]
    result = _compute_oob_push_direction(scene, obj)
    if result is None:
        return
    dx, dz, oob = result
    step = min(0.35, max(0.04, math.sqrt(oob + 1e-8)))
    pos = list(obj.get("pos", [0.0, 0.0, 0.0]))
    obj["pos"] = [pos[0] + dx * step, pos[1], pos[2] + dz * step]


def _separate_local_collisions(scene: Dict[str, Any], idx: int) -> None:
    cols = _collision_neighbors(scene, idx)
    if not cols:
        return
    obj = scene.get("objects", [])[idx]
    total_dx = 0.0
    total_dz = 0.0
    for _, dx, dz, area in cols:
        mag = min(0.28, max(0.02, math.sqrt(area + 1e-8)))
        total_dx += dx * mag
        total_dz += dz * mag
    pos = list(obj.get("pos", [0.0, 0.0, 0.0]))
    obj["pos"] = [pos[0] + total_dx, pos[1], pos[2] + total_dz]
    _project_inside_room(scene, idx)


def _repair_object_local(scene: Dict[str, Any], idx: int, passes: int) -> None:
    objs = scene.get("objects", [])
    if not (0 <= idx < len(objs)):
        return
    for _ in range(max(1, passes)):
        _project_inside_room(scene, idx)
        _separate_local_collisions(scene, idx)


# ------------------------------
# relation + functional losses
# ------------------------------

def _loss_near(d: float, target: float, tol: float, weight: float) -> float:
    return 0.0 if d <= target + tol else (d - target - tol) * weight


def _loss_distance_band(d: float, lo: float, hi: float, weight: float) -> float:
    if d < lo:
        return (lo - d) * weight
    if d > hi:
        return (d - hi) * weight
    return 0.0


def _loss_facing(src_obj: Dict[str, Any], tgt_obj: Dict[str, Any], weight: float) -> float:
    spos = src_obj.get("pos", [0.0, 0.0, 0.0])
    tpos = tgt_obj.get("pos", [0.0, 0.0, 0.0])
    yaw = yaw_from_quaternion(src_obj.get("rot", [0.0, 0.0, 0.0, 1.0]))
    fx, fz = _forward_vec_from_yaw(yaw)
    tx, tz = tpos[0] - spos[0], tpos[2] - spos[2]
    norm = math.hypot(tx, tz)
    if norm < 1e-9:
        return 0.0
    tx, tz = tx / norm, tz / norm
    return max(0.0, 1.0 - (fx * tx + fz * tz)) * weight


def _loss_centered_lateral(src_obj: Dict[str, Any], anchor_obj: Dict[str, Any], weight: float) -> float:
    apos = anchor_obj.get("pos", [0.0, 0.0, 0.0])
    spos = src_obj.get("pos", [0.0, 0.0, 0.0])
    ayaw = yaw_from_quaternion(anchor_obj.get("rot", [0.0, 0.0, 0.0, 1.0]))
    fx, fz = _forward_vec_from_yaw(ayaw)
    lx, lz = fz, -fx
    lateral = abs(_signed_proj(apos[0], apos[2], spos[0], spos[2], lx, lz))
    return lateral * weight


def _loss_in_front_of(src_obj: Dict[str, Any], anchor_obj: Dict[str, Any], weight: float) -> float:
    apos = anchor_obj.get("pos", [0.0, 0.0, 0.0])
    spos = src_obj.get("pos", [0.0, 0.0, 0.0])
    ayaw = yaw_from_quaternion(anchor_obj.get("rot", [0.0, 0.0, 0.0, 1.0]))
    fx, fz = _forward_vec_from_yaw(ayaw)
    proj = _signed_proj(apos[0], apos[2], spos[0], spos[2], fx, fz)
    return 0.0 if proj >= 0 else abs(proj) * weight


def _loss_side_of(src_obj: Dict[str, Any], anchor_obj: Dict[str, Any], weight: float) -> float:
    apos = anchor_obj.get("pos", [0.0, 0.0, 0.0])
    spos = src_obj.get("pos", [0.0, 0.0, 0.0])
    ayaw = yaw_from_quaternion(anchor_obj.get("rot", [0.0, 0.0, 0.0, 1.0]))
    fx, fz = _forward_vec_from_yaw(ayaw)
    lx, lz = fz, -fx
    forward = abs(_signed_proj(apos[0], apos[2], spos[0], spos[2], fx, fz))
    lateral = abs(_signed_proj(apos[0], apos[2], spos[0], spos[2], lx, lz))
    loss = 0.0
    if lateral < 0.2:
        loss += 0.2 - lateral
    if forward > 0.7:
        loss += forward - 0.7
    return loss * weight


def _loss_against_wall(scene: Dict[str, Any], obj: Dict[str, Any], weight: float, *, category: Optional[str] = None) -> float:
    pos = obj.get("pos", [0.0, 0.0, 0.0])
    wall_yaw = _find_nearest_wall_yaw(scene, pos)
    if wall_yaw is None:
        return 0.0
    dist_loss = max(0.0, distance_to_nearest_wall_xz(scene, pos) - (0.35 if category == "bed" else 0.3))
    yaw = yaw_from_quaternion(obj.get("rot", [0.0, 0.0, 0.0, 1.0]))
    if category == "bed":
        yaw_loss = _axis_angle_diff(yaw, wall_yaw) / 180.0
    elif category in {"table", "desk", "counter", "vanity", "cabinet", "console table"}:
        parallel = _nearest_parallel_wall_yaw(scene, pos, yaw)
        yaw_loss = 0.0 if parallel is None else _angle_diff(yaw, parallel) / 180.0
    else:
        yaw_loss = _angle_diff(yaw, _normalize_angle(wall_yaw + 180.0)) / 180.0
    return (dist_loss + 0.5 * yaw_loss) * weight


def _loss_parallel(scene: Dict[str, Any], obj: Dict[str, Any], weight: float) -> float:
    pos = obj.get("pos", [0.0, 0.0, 0.0])
    yaw = yaw_from_quaternion(obj.get("rot", [0.0, 0.0, 0.0, 1.0]))
    parallel = _nearest_parallel_wall_yaw(scene, pos, yaw)
    if parallel is None:
        return 0.0
    diff = _axis_angle_diff(yaw, parallel)
    return (diff / 180.0) * weight


def _compute_relation_loss(scene: Dict[str, Any], role_graph: RoleGraph, priors: List[Dict[str, Any]]) -> Tuple[float, List[Dict[str, Any]]]:
    objs = scene.get("objects", [])
    total = 0.0
    violations: List[Dict[str, Any]] = []

    for item in priors:
        rel_type = str(item.get("type", "")).strip()
        if rel_type not in REL_TYPES:
            continue
        src_idx = item.get("src_idx")
        tgt_idx = item.get("tgt_idx")
        weight = float(item.get("weight", 1.0)) * max(0.0, min(1.0, float(item.get("confidence", 1.0))))

        if not isinstance(src_idx, int) or not (0 <= src_idx < len(objs)):
            continue
        src = objs[src_idx]
        loss = 0.0
        dist: Optional[float] = None

        if rel_type == "against_wall":
            loss = _loss_against_wall(scene, src, weight, category=role_graph.categories[src_idx])
        elif rel_type == "parallel":
            loss = _loss_parallel(scene, src, weight)
        else:
            if not isinstance(tgt_idx, int) or not (0 <= tgt_idx < len(objs)) or tgt_idx == src_idx:
                continue
            tgt = objs[tgt_idx]
            dist = xz_dist(src, tgt)

            if rel_type == "near":
                loss = _loss_near(dist, _pair_target_dist(src, tgt), tol=0.25, weight=weight)
            elif rel_type == "distance_band":
                center = _pair_target_dist(src, tgt, alpha=0.5, bias=0.4)
                loss = _loss_distance_band(dist, max(0.3, center - 0.5), center + 0.8, weight)
            elif rel_type == "facing":
                loss = _loss_facing(src, tgt, weight)
            elif rel_type == "facing_pair":
                loss = 0.5 * _loss_facing(src, tgt, weight) + 0.5 * _loss_facing(tgt, src, weight)
            elif rel_type == "centered_with":
                loss = _loss_centered_lateral(src, tgt, weight)
            elif rel_type == "in_front_of":
                loss = _loss_in_front_of(src, tgt, weight)
            elif rel_type == "side_of":
                loss = _loss_side_of(src, tgt, weight)

        if loss > 1e-8:
            total += loss
            violations.append({"src_idx": src_idx, "tgt_idx": tgt_idx, "type": rel_type, "dist": None if dist is None else round(dist, 3), "penalty": round(loss, 4)})

    return total, violations


# ------------------------------
# scoring
# ------------------------------

_W_PBL = 0.50
_W_REL = 0.20
_W_FUNC = 0.30
_W_SPREAD = 0.18
_W_FLOW = 0.10
_W_MONO = 0.12
_W_OPEN = 0.10


def _get_float_metric(metrics: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else default


def _is_valid_pbl_value(pbl: float, cfg: Config) -> bool:
    return pbl <= cfg.valid_pbl_threshold


def _object_xz_polygon(obj: Dict[str, Any]) -> ShapelyPolygon:
    try:
        poly, _, _, _ = get_xz_bbox_from_obj(obj)
        return poly
    except Exception:
        pos = obj.get("pos", [0.0, 0.0, 0.0])
        return ShapelyPoint(float(pos[0]), float(pos[2])).buffer(0.05)


def _room_extents_xz(scene: Dict[str, Any]) -> Tuple[float, float, float, float]:
    pts = [(float(p[0]), float(p[2])) for p in scene.get("bounds_bottom", []) if isinstance(p, list) and len(p) >= 3]
    if not pts:
        return -1.0, 1.0, -1.0, 1.0
    xs = [p[0] for p in pts]
    zs = [p[1] for p in pts]
    return min(xs), max(xs), min(zs), max(zs)


def _collect_zone_groups(role_graph: RoleGraph, n_obj: int) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for i in range(n_obj):
        label = role_graph.zone_by_idx.get(i) or role_graph.function_by_idx.get(i) or role_graph.role_by_idx.get(i) or f"zone_{i}"
        label = str(label)
        groups.setdefault(label, []).append(i)
    return groups


def _compute_structure_stats(scene: Dict[str, Any], role_graph: RoleGraph, cfg: Optional[Config] = None) -> Dict[str, float]:
    objs = scene.get("objects", [])
    floor_polygon = _get_floor_polygon(scene)
    minx, maxx, minz, maxz = _room_extents_xz(scene)
    room_area = float(floor_polygon.area) if floor_polygon is not None else max(1e-6, (maxx - minx) * (maxz - minz))
    room_w = max(1e-6, maxx - minx)
    room_h = max(1e-6, maxz - minz)

    polys = [_object_xz_polygon(obj) for obj in objs]
    union_poly = unary_union(polys) if polys else None
    occupied_area = 0.0 if union_poly is None else min(room_area, float(union_poly.area))
    open_space_ratio = max(0.0, 1.0 - occupied_area / max(room_area, 1e-6))

    groups = _collect_zone_groups(role_graph, len(objs))
    zone_count = len([idxs for idxs in groups.values() if idxs])
    max_zone_ratio = 1.0 if len(objs) == 0 else max((len(idxs) / len(objs) for idxs in groups.values()), default=1.0)

    spread_penalty = 0.0
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        xs = [float(objs[i].get("pos", [0.0, 0.0, 0.0])[0]) for i in idxs]
        zs = [float(objs[i].get("pos", [0.0, 0.0, 0.0])[2]) for i in idxs]
        cluster_area = max(1e-4, (max(xs) - min(xs) + 0.05) * (max(zs) - min(zs) + 0.05))
        footprint_sum = sum(float(_object_xz_polygon(objs[i]).area) for i in idxs)
        desired_area = 0.35 * footprint_sum + 0.08 * max(0, len(idxs) - 2)
        spread_penalty += max(0.0, desired_area - cluster_area) / max(room_area, 1e-6)

    flow_penalty = 0.0
    if floor_polygon is not None and union_poly is not None:
        cx, cz = room_center_xz(scene)
        corridor_ratio = 0.18 if cfg is None else float(cfg.corridor_width_ratio)
        if room_w >= room_h:
            half = corridor_ratio * room_h * 0.5
            corridor = ShapelyPolygon([(minx, cz - half), (maxx, cz - half), (maxx, cz + half), (minx, cz + half)])
        else:
            half = corridor_ratio * room_w * 0.5
            corridor = ShapelyPolygon([(cx - half, minz), (cx + half, minz), (cx + half, maxz), (cx - half, maxz)])
        corridor = corridor.intersection(floor_polygon)
        if corridor is not None and not corridor.is_empty and corridor.area > 1e-6:
            overlap_ratio = float(union_poly.intersection(corridor).area) / float(corridor.area)
            flow_penalty = max(0.0, overlap_ratio - 0.22)

    mono_target = 0.72 if cfg is None else float(cfg.max_zone_monopoly_ratio)
    open_target = 0.42 if cfg is None else float(cfg.min_open_space_ratio)
    monopoly_penalty = max(0.0, max_zone_ratio - mono_target)
    open_penalty = max(0.0, open_target - open_space_ratio)
    structure_loss = _W_SPREAD * spread_penalty + _W_FLOW * flow_penalty + _W_MONO * monopoly_penalty + _W_OPEN * open_penalty

    return {
        "room_area": round(room_area, 6),
        "occupied_area": round(occupied_area, 6),
        "open_space_ratio": round(open_space_ratio, 6),
        "zone_count": float(zone_count),
        "max_zone_ratio": round(max_zone_ratio, 6),
        "spread_penalty": round(spread_penalty, 6),
        "flow_penalty": round(flow_penalty, 6),
        "monopoly_penalty": round(monopoly_penalty, 6),
        "open_penalty": round(open_penalty, 6),
        "structure_loss": round(structure_loss, 6),
    }


def _score_scene_full(
    scene: Dict[str, Any],
    role_graph: Optional[RoleGraph] = None,
    relation_priors: Optional[List[Dict[str, Any]]] = None,
    timing_stats: Optional[TimingStats] = None,
    cfg: Optional[Config] = None,
) -> Tuple[float, Dict[str, Any], float, float]:
    t0 = time.perf_counter()
    metrics = eval_scene(scene, is_debug=False)
    if role_graph is None:
        role_graph = infer_role_graph(scene)
    priors = relation_priors or []
    rel_loss, _ = _compute_relation_loss(scene, role_graph, priors)
    func_loss = compute_functional_loss(scene, role_graph, yaw_from_quaternion).total
    metrics = dict(metrics)
    structure_stats = _compute_structure_stats(scene, role_graph, cfg)
    metrics["structure_stats"] = structure_stats
    metrics["open_space_ratio"] = structure_stats["open_space_ratio"]
    metrics["zone_count"] = int(structure_stats["zone_count"])
    metrics["max_zone_ratio"] = structure_stats["max_zone_ratio"]
    metrics["structure_loss"] = structure_stats["structure_loss"]
    score = (
        _W_PBL * _get_float_metric(metrics, "total_pbl_loss")
        + _W_REL * rel_loss
        + _W_FUNC * func_loss
        + structure_stats["structure_loss"]
    )
    if timing_stats is not None:
        timing_stats.eval_sec += time.perf_counter() - t0
    return score, metrics, rel_loss, func_loss


def _collision_area_for_object(scene: Dict[str, Any], idx: int) -> float:
    return sum(area for _, _, _, area in _collision_neighbors(scene, idx))


def _relation_penalty_for_object(scene: Dict[str, Any], role_graph: RoleGraph, priors: List[Dict[str, Any]], idx: int) -> float:
    _, violations = _compute_relation_loss(scene, role_graph, priors)
    return sum(v["penalty"] for v in violations if v.get("src_idx") == idx or v.get("tgt_idx") == idx)


def _functional_penalty_for_object(scene: Dict[str, Any], role_graph: RoleGraph, idx: int) -> float:
    violations = compute_functional_loss(scene, role_graph, yaw_from_quaternion).violations
    total = 0.0
    for item in violations:
        if item.get("idx") == idx or item.get("anchor_idx") == idx or item.get("src_idx") == idx or item.get("tgt_idx") == idx:
            total += float(item.get("penalty", 0.0))
    return total


def _quick_candidate_proxy_score(scene: Dict[str, Any], role_graph: RoleGraph, priors: List[Dict[str, Any]], idx: int, cfg: Optional[Config] = None) -> float:
    floor_polygon = _get_floor_polygon(scene)
    oob = 0.0
    if floor_polygon is not None:
        oob = compute_oob(scene["objects"][idx], floor_polygon, scene.get("bounds_bottom", []), scene.get("bounds_top", []), is_debug=False)
    collision = _collision_area_for_object(scene, idx)
    relation = _relation_penalty_for_object(scene, role_graph, priors, idx)
    functional = _functional_penalty_for_object(scene, role_graph, idx)
    structure = _compute_structure_stats(scene, role_graph, cfg)
    return 3.0 * oob + 2.0 * collision + 1.2 * relation + 1.4 * functional + 1.1 * float(structure["spread_penalty"]) + 0.8 * float(structure["flow_penalty"]) + 1.1 * float(structure["monopoly_penalty"])


# ------------------------------
# moves and candidates
# ------------------------------

def _find_obj_idx_for_edit(scene: Dict[str, Any], edit: ObjectEdit) -> Optional[int]:
    best_idx = None
    best_dist = float("inf")
    for i, obj in enumerate(scene.get("objects", [])):
        jid = obj.get("sampled_asset_jid") or obj.get("jid") or obj.get("sampled_jid") or ""
        if not isinstance(jid, str) or len(jid) < 6:
            continue
        if jid[:6].lower() != edit.jid_prefix.lower():
            continue
        if edit.hint_pos is None:
            return i
        pos = obj.get("pos", [0.0, 0.0, 0.0])
        d = math.hypot(float(pos[0]) - float(edit.hint_pos[0]), float(pos[2]) - float(edit.hint_pos[2]))
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def _apply_delta(scene: Dict[str, Any], idx: int, dx: float = 0.0, dz: float = 0.0, dyaw: float = 0.0, yaw_abs: Optional[float] = None) -> Dict[str, Any]:
    obj = copy.deepcopy(scene["objects"][idx])
    pos = list(obj.get("pos", [0.0, 0.0, 0.0]))
    pos[0] += dx
    pos[2] += dz
    obj["pos"] = pos
    if yaw_abs is not None:
        obj["rot"] = quaternion_from_yaw(_normalize_angle(yaw_abs))
    elif abs(dyaw) > 1e-9:
        yaw = yaw_from_quaternion(obj.get("rot", [0.0, 0.0, 0.0, 1.0]))
        obj["rot"] = quaternion_from_yaw(_normalize_angle(yaw + dyaw))
    return _clone_scene_with_updated_object(scene, idx, obj)


def _anchor_pose_candidate(scene: Dict[str, Any], role_graph: RoleGraph, idx: int) -> Optional[Dict[str, Any]]:
    objs = scene.get("objects", [])
    cat = role_graph.categories[idx]
    obj = objs[idx]
    pos = obj.get("pos", [0.0, 0.0, 0.0])

    if idx in role_graph.accessory_to_anchor:
        anchor_idx = role_graph.accessory_to_anchor[idx]
        objs[idx]["_category"] = cat
        objs[anchor_idx]["_category"] = role_graph.categories[anchor_idx]
        target_pos, target_yaw = target_pose_for_attachment(scene, objs, idx, anchor_idx, yaw_from_quaternion)
        return {"kind": "role_attachment", "dx": target_pos[0] - pos[0], "dz": target_pos[2] - pos[2], "yaw_abs": target_yaw, "dyaw": 0.0}

    if cat == "bed":
        yaw_abs = _nearest_normal_axis_yaw(scene, pos, yaw_from_quaternion(obj.get("rot", [0.0, 0.0, 0.0, 1.0])))
        if yaw_abs is not None:
            return {"kind": "bed_align_wall", "dx": 0.0, "dz": 0.0, "yaw_abs": yaw_abs, "dyaw": 0.0}

    if idx in role_graph.parallel_wall_indices:
        yaw_abs = _nearest_parallel_wall_yaw(scene, pos, yaw_from_quaternion(obj.get("rot", [0.0, 0.0, 0.0, 1.0])))
        if yaw_abs is not None:
            return {"kind": "parallel_wall", "dx": 0.0, "dz": 0.0, "yaw_abs": yaw_abs, "dyaw": 0.0}

    if idx in role_graph.wall_affine_indices:
        wall_yaw = _find_nearest_wall_yaw(scene, pos)
        if wall_yaw is not None:
            return {"kind": "against_wall", "dx": 0.0, "dz": 0.0, "yaw_abs": _normalize_angle(wall_yaw + 180.0), "dyaw": 0.0}

    return None


def _zone_release_candidates(scene: Dict[str, Any], role_graph: RoleGraph, idx: int, cfg: Config) -> List[Dict[str, Any]]:
    if not cfg.add_zone_release_candidates:
        return []
    objs = scene.get("objects", [])
    if not (0 <= idx < len(objs)) or len(objs) < 4:
        return []
    groups = _collect_zone_groups(role_graph, len(objs))
    if not groups:
        return []
    dominant_label, dominant_indices = max(groups.items(), key=lambda kv: len(kv[1]))
    if idx not in dominant_indices:
        return []
    if len(dominant_indices) / max(1, len(objs)) <= cfg.max_zone_monopoly_ratio:
        return []
    if idx in role_graph.dominant_indices or idx in set(role_graph.accessory_to_anchor.values()):
        return []
    cat = role_graph.categories[idx]
    if cat not in {"chair", "lamp", "table", "coffee table", "sofa", "bench", "plant", "nightstand"}:
        return []

    dom_x = sum(float(objs[i].get("pos", [0.0, 0.0, 0.0])[0]) for i in dominant_indices) / max(1, len(dominant_indices))
    dom_z = sum(float(objs[i].get("pos", [0.0, 0.0, 0.0])[2]) for i in dominant_indices) / max(1, len(dominant_indices))
    minx, maxx, minz, maxz = _room_extents_xz(scene)
    inset_x = (maxx - minx) * cfg.zone_release_inset_ratio
    inset_z = (maxz - minz) * cfg.zone_release_inset_ratio
    anchors = [
        (minx + inset_x, minz + inset_z),
        (minx + inset_x, maxz - inset_z),
        (maxx - inset_x, minz + inset_z),
        (maxx - inset_x, maxz - inset_z),
        (minx + inset_x, (minz + maxz) * 0.5),
        (maxx - inset_x, (minz + maxz) * 0.5),
        ((minx + maxx) * 0.5, minz + inset_z),
        ((minx + maxx) * 0.5, maxz - inset_z),
    ]
    cur = objs[idx].get("pos", [0.0, 0.0, 0.0])
    others = [j for j in range(len(objs)) if j != idx]
    scored: List[Tuple[float, Tuple[float, float]]] = []
    for tx, tz in anchors:
        dist_from_dom = math.hypot(tx - dom_x, tz - dom_z)
        nearest_other = min((math.hypot(tx - float(objs[j].get("pos", [0.0, 0.0, 0.0])[0]), tz - float(objs[j].get("pos", [0.0, 0.0, 0.0])[2])) for j in others), default=1.0)
        move_cost = math.hypot(tx - float(cur[0]), tz - float(cur[2]))
        score = 1.2 * dist_from_dom + 0.8 * nearest_other - 0.35 * move_cost
        scored.append((score, (tx, tz)))
    scored.sort(reverse=True)
    results: List[Dict[str, Any]] = []
    for _, (tx, tz) in scored[:2]:
        yaw_abs = None
        if cat in {"chair", "sofa", "bench"}:
            yaw_abs = _normalize_angle(math.degrees(math.atan2(dom_x - tx, dom_z - tz)))
        elif idx in role_graph.parallel_wall_indices:
            yaw_abs = _nearest_parallel_wall_yaw(scene, [tx, 0.0, tz], yaw_from_quaternion(objs[idx].get("rot", [0.0, 0.0, 0.0, 1.0])))
        results.append({
            "kind": "zone_release",
            "dx": float(tx - float(cur[0])),
            "dz": float(tz - float(cur[2])),
            "dyaw": 0.0,
            "yaw_abs": yaw_abs,
        })
    return results


def _dominant_zone_release_indices(scene: Dict[str, Any], role_graph: RoleGraph, cfg: Config) -> List[int]:
    objs = scene.get("objects", [])
    groups = _collect_zone_groups(role_graph, len(objs))
    if not groups:
        return []
    dominant_label, dominant_indices = max(groups.items(), key=lambda kv: len(kv[1]))
    if len(dominant_indices) / max(1, len(objs)) <= max(cfg.max_zone_monopoly_ratio, cfg.zone_layout_trigger_ratio):
        return []
    dominant_anchor_values = set(role_graph.accessory_to_anchor.values())
    priority = {
        "coffee table": 0,
        "table": 1,
        "chair": 2,
        "bench": 3,
        "lamp": 4,
        "sofa": 5,
        "plant": 6,
        "nightstand": 7,
    }
    dom_x = sum(float(objs[i].get("pos", [0.0, 0.0, 0.0])[0]) for i in dominant_indices) / max(1, len(dominant_indices))
    dom_z = sum(float(objs[i].get("pos", [0.0, 0.0, 0.0])[2]) for i in dominant_indices) / max(1, len(dominant_indices))
    releasable = []
    for i in dominant_indices:
        if i in role_graph.dominant_indices or i in dominant_anchor_values:
            continue
        cat = role_graph.categories[i]
        if cat not in priority:
            continue
        pos = objs[i].get("pos", [0.0, 0.0, 0.0])
        d = math.hypot(float(pos[0]) - dom_x, float(pos[2]) - dom_z)
        releasable.append((priority[cat], d, i))
    releasable.sort(key=lambda x: (x[0], x[1]))
    return [i for _, _, i in releasable[: max(1, cfg.zone_layout_topk_objects)]]


def _zone_layout_anchor_points(scene: Dict[str, Any], cfg: Config) -> List[Tuple[float, float]]:
    minx, maxx, minz, maxz = _room_extents_xz(scene)
    inset_x = (maxx - minx) * cfg.zone_layout_anchor_inset_ratio
    inset_z = (maxz - minz) * cfg.zone_layout_anchor_inset_ratio
    return [
        (minx + inset_x, minz + inset_z),
        (minx + inset_x, maxz - inset_z),
        (maxx - inset_x, minz + inset_z),
        (maxx - inset_x, maxz - inset_z),
        (minx + inset_x, 0.5 * (minz + maxz)),
        (maxx - inset_x, 0.5 * (minz + maxz)),
        (0.5 * (minx + maxx), minz + inset_z),
        (0.5 * (minx + maxx), maxz - inset_z),
    ]


def _best_wall_aligned_yaw(scene: Dict[str, Any], obj: Dict[str, Any], cat: str, tx: float, tz: float) -> Optional[float]:
    current_yaw = yaw_from_quaternion(obj.get("rot", [0.0, 0.0, 0.0, 1.0]))
    if cat == "bed":
        return _nearest_normal_axis_yaw(scene, [tx, 0.0, tz], current_yaw)
    if cat in {"table", "coffee table", "desk", "counter", "vanity", "cabinet", "console table"}:
        return _nearest_parallel_wall_yaw(scene, [tx, 0.0, tz], current_yaw)
    if cat in {"wardrobe", "shelf", "tv stand", "refrigerator", "washing machine", "dryer"}:
        wall_yaw = _find_nearest_wall_yaw(scene, [tx, 0.0, tz])
        return None if wall_yaw is None else _normalize_angle(wall_yaw + 180.0)
    return None


def _build_zone_layout_macro_candidates(scene: Dict[str, Any], role_graph: RoleGraph, metrics: Dict[str, Any], cfg: Config) -> List[Dict[str, Any]]:
    if not cfg.enable_zone_layout_candidates:
        return []
    struct = metrics.get("structure_stats", {}) if isinstance(metrics, dict) else {}
    objs = scene.get("objects", [])
    if len(objs) < 4:
        return []
    zone_count = int(float(struct.get("zone_count", 0.0)))
    max_zone_ratio = float(struct.get("max_zone_ratio", 0.0))
    open_space_ratio = float(struct.get("open_space_ratio", 0.0))
    if max_zone_ratio <= max(cfg.max_zone_monopoly_ratio, cfg.zone_layout_trigger_ratio) and zone_count >= 2 and open_space_ratio >= cfg.min_open_space_ratio:
        return []

    release_indices = _dominant_zone_release_indices(scene, role_graph, cfg)
    if not release_indices:
        return []

    groups = _collect_zone_groups(role_graph, len(objs))
    dominant_label, dominant_indices = max(groups.items(), key=lambda kv: len(kv[1]))
    dom_x = sum(float(objs[i].get("pos", [0.0, 0.0, 0.0])[0]) for i in dominant_indices) / max(1, len(dominant_indices))
    dom_z = sum(float(objs[i].get("pos", [0.0, 0.0, 0.0])[2]) for i in dominant_indices) / max(1, len(dominant_indices))

    anchors = _zone_layout_anchor_points(scene, cfg)
    scored = []
    for tx, tz in anchors:
        d = math.hypot(tx - dom_x, tz - dom_z)
        scored.append((d, tx, tz))
    scored.sort(reverse=True)

    cat_by_idx = {i: role_graph.categories[i] for i in release_indices}
    support_idx = next((i for i in release_indices if cat_by_idx[i] in {"coffee table", "table", "desk", "bench"}), None)
    seat_like = [i for i in release_indices if cat_by_idx[i] in {"chair", "sofa", "bench"} and i != support_idx]
    deco_like = [i for i in release_indices if i not in seat_like and i != support_idx]

    candidates: List[Dict[str, Any]] = []
    radius = max(0.35, cfg.zone_layout_secondary_cluster_radius)
    for _, tx, tz in scored[: max(1, cfg.max_zone_layout_candidates)]:
        moves: List[Dict[str, Any]] = []
        if support_idx is not None:
            support_obj = objs[support_idx]
            support_cat = cat_by_idx[support_idx]
            support_yaw = _best_wall_aligned_yaw(scene, support_obj, support_cat, tx, tz)
            moves.append({"idx": support_idx, "tx": tx, "tz": tz, "yaw_abs": support_yaw})
            for k, i in enumerate(seat_like[:2]):
                side = -1.0 if k % 2 == 0 else 1.0
                px = tx + side * radius
                pz = tz + (0.18 if k > 1 else 0.0)
                yaw_abs = _normalize_angle(math.degrees(math.atan2(tx - px, tz - pz)))
                moves.append({"idx": i, "tx": px, "tz": pz, "yaw_abs": yaw_abs})
            for k, i in enumerate(deco_like[:2]):
                px = tx + (0.22 + 0.18 * k)
                pz = tz - (0.22 + 0.12 * k)
                yaw_abs = _best_wall_aligned_yaw(scene, objs[i], cat_by_idx[i], px, pz)
                moves.append({"idx": i, "tx": px, "tz": pz, "yaw_abs": yaw_abs})
        else:
            offsets = [(0.0, 0.0), (radius, 0.0), (-radius, 0.0), (0.0, radius), (0.0, -radius)]
            for k, i in enumerate(release_indices):
                ox, oz = offsets[min(k, len(offsets) - 1)]
                px = tx + ox
                pz = tz + oz
                yaw_abs = None
                if cat_by_idx[i] in {"chair", "sofa", "bench"}:
                    yaw_abs = _normalize_angle(math.degrees(math.atan2(tx - px, tz - pz)))
                else:
                    yaw_abs = _best_wall_aligned_yaw(scene, objs[i], cat_by_idx[i], px, pz)
                moves.append({"idx": i, "tx": px, "tz": pz, "yaw_abs": yaw_abs})

        if not moves:
            continue
        candidates.append({
            "kind": "zone_layout_macro",
            "anchor": [round(tx, 4), round(tz, 4)],
            "dominant_label": str(dominant_label),
            "release_indices": list(release_indices),
            "moves": moves,
        })
    return candidates


def _apply_zone_layout_macro(scene: Dict[str, Any], macro: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    sc = _deepcopy_scene(scene)
    for move in macro.get("moves", []):
        idx = int(move["idx"])
        obj = copy.deepcopy(sc["objects"][idx])
        pos = list(obj.get("pos", [0.0, 0.0, 0.0]))
        pos[0] = float(move["tx"])
        pos[2] = float(move["tz"])
        obj["pos"] = pos
        yaw_abs = move.get("yaw_abs")
        if yaw_abs is not None:
            obj["rot"] = quaternion_from_yaw(_normalize_angle(float(yaw_abs)))
        sc = _clone_scene_with_updated_object(sc, idx, obj)
        _repair_object_local(sc, idx, max(1, cfg.local_repair_passes + 1))
    return sc


def _evaluate_best_zone_layout(
    scene: Dict[str, Any],
    role_graph: RoleGraph,
    priors: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    current_score: float,
    cfg: Config,
    *,
    fixed_priors: bool,
    current_pbl: float,
    initial_priors: Optional[List[Dict[str, Any]]] = None,
    vlm_priors: Optional[List[Dict[str, Any]]] = None,
    timing: Optional[TimingStats] = None,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], float, float, float, Dict[str, Any], RoleGraph, List[Dict[str, Any]]]]:
    macros = _build_zone_layout_macro_candidates(scene, role_graph, metrics, cfg)
    best = None
    for cand in macros:
        sc = _apply_zone_layout_macro(scene, cand, cfg)
        local_role_graph = infer_role_graph(sc)
        if fixed_priors:
            local_priors = list(initial_priors or priors)
        else:
            det_priors = build_role_based_relation_priors(sc, local_role_graph)
            local_priors = _merge_relation_priors(sc, local_role_graph, det_priors, vlm_priors, cfg.vlm_prior_weight_scale) if vlm_priors and cfg.merge_vlm_with_deterministic else (list(vlm_priors) if vlm_priors else det_priors)
        score, new_metrics, rel_loss, func_loss = _score_scene_full(sc, local_role_graph, local_priors, timing, cfg)
        next_pbl = _get_float_metric(new_metrics, "total_pbl_loss")
        if cfg.candidate_filter_reintroduced_pbl and _is_valid_pbl_value(current_pbl, cfg) and not _is_valid_pbl_value(next_pbl, cfg):
            continue
        if score < current_score - max(cfg.monotonic_eps, cfg.zone_layout_min_score_improve):
            if best is None or score < best[4]:
                best = (sc, new_metrics, rel_loss, func_loss, score, cand, local_role_graph, local_priors)
    return best


def _generate_candidates(scene: Dict[str, Any], role_graph: RoleGraph, idx: int, bias_edit: Optional[ObjectEdit], cfg: Config) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    obj = scene["objects"][idx]
    cat = role_graph.categories[idx]
    step_xy = cfg.step_xy * max(0.8, min(1.6, obj_diag_size_xz(obj)))
    step_yaw = cfg.step_yaw

    if bias_edit is not None:
        yaw_abs = None if bias_edit.target_yaw_deg is None else float(bias_edit.target_yaw_deg)
        dyaw = 0.0 if bias_edit.no_rotation else float(bias_edit.relative_yaw_deg or 0.0)
        if idx in role_graph.parallel_wall_indices:
            base_yaw = yaw_abs
            if base_yaw is None and abs(dyaw) > 1e-9:
                base_yaw = yaw_from_quaternion(obj.get("rot", [0.0, 0.0, 0.0, 1.0])) + dyaw
                dyaw = 0.0
            if base_yaw is not None:
                snapped = _nearest_parallel_wall_yaw(scene, obj.get("pos", [0.0, 0.0, 0.0]), base_yaw)
                if snapped is not None:
                    yaw_abs = snapped
                    dyaw = 0.0
        elif cat == "bed":
            base_yaw = yaw_abs
            if base_yaw is None and abs(dyaw) > 1e-9:
                base_yaw = yaw_from_quaternion(obj.get("rot", [0.0, 0.0, 0.0, 1.0])) + dyaw
                dyaw = 0.0
            if base_yaw is not None:
                snapped = _nearest_normal_axis_yaw(scene, obj.get("pos", [0.0, 0.0, 0.0]), base_yaw)
                if snapped is not None:
                    yaw_abs = snapped
                    dyaw = 0.0
        candidates.append({"kind": "gpt_bias", "dx": float(bias_edit.dx), "dz": float(bias_edit.dz), "dyaw": dyaw, "yaw_abs": yaw_abs})

    anchor_cand = _anchor_pose_candidate(scene, role_graph, idx)
    if anchor_cand is not None:
        candidates.append(anchor_cand)

    allow_free_yaw = idx not in role_graph.parallel_wall_indices and cat != "bed"
    allow_diag = cat not in {"table", "counter"}
    for sign in (-1.0, 1.0):
        candidates.append({"kind": "axis_x", "dx": sign * step_xy, "dz": 0.0, "dyaw": 0.0, "yaw_abs": None})
        candidates.append({"kind": "axis_z", "dx": 0.0, "dz": sign * step_xy, "dyaw": 0.0, "yaw_abs": None})
        if allow_free_yaw:
            candidates.append({"kind": "axis_yaw", "dx": 0.0, "dz": 0.0, "dyaw": sign * step_yaw, "yaw_abs": None})
        if allow_diag:
            candidates.append({"kind": "diag", "dx": sign * step_xy * 0.7, "dz": sign * step_xy * 0.7, "dyaw": 0.0, "yaw_abs": None})
            candidates.append({"kind": "diag2", "dx": sign * step_xy * 0.7, "dz": -sign * step_xy * 0.7, "dyaw": 0.0, "yaw_abs": None})

    candidates.extend(_zone_release_candidates(scene, role_graph, idx, cfg))

    unique: Dict[Tuple[int, int, int, Optional[int]], Dict[str, Any]] = {}
    for cand in candidates:
        key = (
            round(cand.get("dx", 0.0) * 100),
            round(cand.get("dz", 0.0) * 100),
            round(cand.get("dyaw", 0.0)),
            None if cand.get("yaw_abs") is None else round(float(cand["yaw_abs"])),
        )
        unique[key] = cand
    return list(unique.values())


def _prioritized_object_indices(scene: Dict[str, Any], role_graph: RoleGraph, edits: List[ObjectEdit], metrics: Dict[str, Any], cfg: Config) -> List[int]:
    hotspot = metrics.get("obj_with_highest_pbl_loss", {})
    hotspot_idx = hotspot.get("idx") if isinstance(hotspot, dict) else None
    edited_indices: List[int] = []
    for edit in edits:
        idx = _find_obj_idx_for_edit(scene, edit)
        if idx is not None and idx not in edited_indices:
            edited_indices.append(idx)

    pbl = _get_float_metric(metrics, "total_pbl_loss")
    lock_major = pbl <= cfg.anchor_lock_pbl_threshold
    order = optimization_stage_order(scene, role_graph, edited_indices, hotspot_idx, cfg.max_objects_per_round, lock_major)
    struct = metrics.get("structure_stats", {}) if isinstance(metrics, dict) else {}
    if float(struct.get("max_zone_ratio", 0.0)) <= cfg.max_zone_monopoly_ratio:
        return order
    groups = _collect_zone_groups(role_graph, len(scene.get("objects", [])))
    if not groups:
        return order
    dominant_label, dominant_idxs = max(groups.items(), key=lambda kv: len(kv[1]))
    dominant_anchor_values = set(role_graph.accessory_to_anchor.values())
    release_first = [i for i in order if i in dominant_idxs and i not in role_graph.dominant_indices and i not in dominant_anchor_values]
    tail = [i for i in order if i not in release_first]
    return release_first + tail


# ------------------------------
# relation priors and VLM merge
# ------------------------------

def _dedup_priors(priors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for item in priors:
        key = (item.get("src_idx"), item.get("tgt_idx"), item.get("type"))
        score = float(item.get("confidence", 0.0)) * float(item.get("weight", 1.0))
        old = unique.get(key)
        old_score = float(old.get("confidence", 0.0)) * float(old.get("weight", 1.0)) if old else -1.0
        if old is None or score > old_score:
            unique[key] = item
    return list(unique.values())


def _merge_relation_priors(
    scene: Dict[str, Any],
    role_graph: RoleGraph,
    deterministic_priors: List[Dict[str, Any]],
    vlm_priors: Optional[List[Dict[str, Any]]],
    weight_scale: float,
) -> List[Dict[str, Any]]:
    if not vlm_priors:
        return deterministic_priors

    merged = list(deterministic_priors)
    for item in vlm_priors:
        if not isinstance(item, dict):
            continue
        src_idx = item.get("src_idx")
        tgt_idx = item.get("tgt_idx")
        rel_type = str(item.get("type", "")).strip()
        if rel_type not in REL_TYPES:
            continue
        if not isinstance(src_idx, int) or not (0 <= src_idx < len(role_graph.categories)):
            continue
        if rel_type not in {"against_wall", "parallel"}:
            if not isinstance(tgt_idx, int) or not (0 <= tgt_idx < len(role_graph.categories)) or tgt_idx == src_idx:
                continue
        src_cat = role_graph.categories[src_idx]
        tgt_cat = role_graph.categories[tgt_idx] if isinstance(tgt_idx, int) and 0 <= tgt_idx < len(role_graph.categories) else "wall"

        if src_idx in role_graph.accessory_to_anchor:
            assigned = role_graph.accessory_to_anchor[src_idx]
            if rel_type not in {"against_wall", "parallel"} and tgt_idx != assigned:
                continue
            if src_cat == "chair" and rel_type == "facing" and role_graph.categories[assigned] in {"desk", "vanity", "counter", "sink"}:
                continue
        if src_cat == "bed" and rel_type in {"facing", "facing_pair", "in_front_of"}:
            continue
        if src_cat in {"table", "desk", "counter", "vanity"} and rel_type in {"facing", "facing_pair"}:
            continue
        if src_cat in {"nightstand"} and tgt_cat not in {"bed", "wall"}:
            continue

        merged.append(
            {
                "src_idx": src_idx,
                "tgt_idx": tgt_idx if rel_type not in {"against_wall", "parallel"} else None,
                "type": rel_type,
                "confidence": min(0.9, float(item.get("confidence", 1.0))),
                "weight": float(item.get("weight", 1.0)) * weight_scale,
                "reason": str(item.get("reason", "")),
            }
        )
    return _dedup_priors(merged)


def _clean_relation_priors(priors: List[Dict[str, Any]], n_obj: int, min_confidence: float) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    per_src_count: Dict[int, int] = {}
    for item in priors:
        if not isinstance(item, dict):
            continue
        rel_type = str(item.get("type", "")).strip()
        if rel_type not in REL_TYPES:
            continue
        src_idx = item.get("src_idx")
        if not isinstance(src_idx, int) or not (0 <= src_idx < n_obj):
            continue
        tgt_idx = item.get("tgt_idx")
        if rel_type not in {"against_wall", "parallel"}:
            if not isinstance(tgt_idx, int) or not (0 <= tgt_idx < n_obj) or tgt_idx == src_idx:
                continue
        else:
            tgt_idx = None
        confidence = float(item.get("confidence", 1.0))
        if confidence < min_confidence:
            continue
        if per_src_count.get(src_idx, 0) >= 3:
            continue
        cleaned.append({
            "src_idx": src_idx,
            "tgt_idx": tgt_idx,
            "type": rel_type,
            "confidence": confidence,
            "weight": float(item.get("weight", 1.0)),
            "reason": str(item.get("reason", "")),
        })
        per_src_count[src_idx] = per_src_count.get(src_idx, 0) + 1
        if len(cleaned) >= 32:
            break
    return cleaned


def _safe_generate_move_prompt(generator: GPTVLMovePromptGeneratorV5, diag_path: Path, top_path: Path, scene: Dict[str, Any], extra_context: str, retries: int, temperature: float, max_tokens: int, trial_dir: Path):
    last_exc: Optional[BaseException] = None
    for retry in range(retries):
        try:
            return generator.generate(diag_image_path=diag_path, top_image_path=top_path, scene=scene, extra_context=extra_context, temperature=temperature, max_tokens=max_tokens)
        except Exception as exc:
            last_exc = exc
            _write_text(trial_dir / f"move_prompt_retry{retry+1}.error.txt", traceback.format_exc())
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("move prompt generation failed")


def _safe_generate_relation_priors(generator: GPTVLMovePromptGeneratorV5, diag_path: Path, top_path: Path, scene: Dict[str, Any], extra_context: str, retries: int, temperature: float, max_tokens: int, min_confidence: float, out_dir: Path) -> Optional[List[Dict[str, Any]]]:
    if not hasattr(generator, "generate_relation_priors"):
        _write_text(out_dir / "vlm_relation_priors.error.txt", "generator does not implement generate_relation_priors(); fallback to deterministic relation priors.\n")
        return None

    last_exc: Optional[BaseException] = None
    for retry in range(retries):
        try:
            result = generator.generate_relation_priors(diag_image_path=diag_path, top_image_path=top_path, scene=scene, extra_context=extra_context, temperature=temperature, max_tokens=max_tokens)
            raw = json.loads(result.json_text)
            cleaned = _clean_relation_priors(raw.get("relations", []), len(scene.get("objects", [])), min_confidence)
            _write_json(out_dir / "vlm_relation_priors.json", {"raw_text": result.raw_text, "raw_json": raw, "cleaned_relations": cleaned})
            return cleaned
        except Exception as exc:
            last_exc = exc
            _write_text(out_dir / f"vlm_relation_priors_retry{retry+1}.error.txt", traceback.format_exc())
    if last_exc is not None:
        return None
    return None


# ------------------------------
# optimization per step
# ------------------------------

def _evaluate_best_local_move(
    scene: Dict[str, Any],
    role_graph: RoleGraph,
    priors: List[Dict[str, Any]],
    idx: int,
    bias_edit: Optional[ObjectEdit],
    current_score: float,
    cfg: Config,
    *,
    fixed_priors: bool,
    current_pbl: float,
    timing: Optional[TimingStats] = None,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], float, float, float, Dict[str, Any], RoleGraph, List[Dict[str, Any]]]]:
    candidates = _generate_candidates(scene, role_graph, idx, bias_edit, cfg)
    proxy_ranked: List[Tuple[float, Dict[str, Any], Dict[str, Any], RoleGraph]] = []

    for cand in candidates:
        sc = _apply_delta(scene, idx, dx=float(cand.get("dx", 0.0)), dz=float(cand.get("dz", 0.0)), dyaw=float(cand.get("dyaw", 0.0)), yaw_abs=None if cand.get("yaw_abs") is None else float(cand.get("yaw_abs")))
        _repair_object_local(sc, idx, cfg.local_repair_passes)
        local_role_graph = infer_role_graph(sc)
        local_priors = priors if fixed_priors else build_role_based_relation_priors(sc, local_role_graph)
        proxy = _quick_candidate_proxy_score(sc, local_role_graph, local_priors, idx, cfg)
        proxy_ranked.append((proxy, cand, sc, local_role_graph))

    proxy_ranked.sort(key=lambda x: x[0])
    best = None
    for _, cand, sc, local_role_graph in proxy_ranked[: max(1, cfg.proxy_topk)]:
        local_priors = priors if fixed_priors else build_role_based_relation_priors(sc, local_role_graph)
        score, metrics, rel_loss, func_loss = _score_scene_full(sc, local_role_graph, local_priors, timing, cfg)
        next_pbl = _get_float_metric(metrics, "total_pbl_loss")
        if (
            cfg.candidate_filter_reintroduced_pbl
            and _is_valid_pbl_value(current_pbl, cfg)
            and not _is_valid_pbl_value(next_pbl, cfg)
        ):
            continue
        if score < current_score - cfg.monotonic_eps:
            if best is None or score < best[4]:
                best = (sc, metrics, rel_loss, func_loss, score, cand, local_role_graph, local_priors)
    return best


def _optimize_after_prompt(
    scene: Dict[str, Any],
    edits: List[ObjectEdit],
    cfg: Config,
    *,
    initial_priors: Optional[List[Dict[str, Any]]] = None,
    fixed_priors: bool = False,
    vlm_priors: Optional[List[Dict[str, Any]]] = None,
    timing: Optional[TimingStats] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], float, float, List[Dict[str, Any]], List[Dict[str, Any]], RoleGraph]:
    cur_scene = _deepcopy_scene(scene)
    role_graph = infer_role_graph(cur_scene)
    if fixed_priors:
        priors = list(initial_priors or [])
    else:
        det_priors = build_role_based_relation_priors(cur_scene, role_graph)
        priors = _merge_relation_priors(cur_scene, role_graph, det_priors, vlm_priors, cfg.vlm_prior_weight_scale) if vlm_priors and cfg.merge_vlm_with_deterministic else (list(vlm_priors) if vlm_priors else det_priors)
    cur_score, cur_metrics, cur_rel, cur_func = _score_scene_full(cur_scene, role_graph, priors, timing, cfg)
    actions: List[Dict[str, Any]] = []

    edit_by_idx: Dict[int, ObjectEdit] = {}
    for edit in edits:
        idx = _find_obj_idx_for_edit(cur_scene, edit)
        if idx is not None and idx not in edit_by_idx:
            edit_by_idx[idx] = edit

    for round_idx in range(cfg.max_rounds):
        improved = False
        role_graph = infer_role_graph(cur_scene)
        if fixed_priors:
            priors = list(initial_priors or [])
        else:
            det_priors = build_role_based_relation_priors(cur_scene, role_graph)
            priors = _merge_relation_priors(cur_scene, role_graph, det_priors, vlm_priors, cfg.vlm_prior_weight_scale) if vlm_priors and cfg.merge_vlm_with_deterministic else (list(vlm_priors) if vlm_priors else det_priors)
        order = _prioritized_object_indices(cur_scene, role_graph, edits, cur_metrics, cfg)

        macro_best = _evaluate_best_zone_layout(
            cur_scene,
            role_graph,
            priors,
            cur_metrics,
            cur_score,
            cfg,
            fixed_priors=fixed_priors,
            current_pbl=_get_float_metric(cur_metrics, "total_pbl_loss"),
            initial_priors=initial_priors,
            vlm_priors=vlm_priors,
            timing=timing,
        )
        if macro_best is not None:
            next_scene, next_metrics, next_rel, next_func, next_score, cand, next_role_graph, next_priors = macro_best
            actions.append({
                "round": round_idx,
                "obj_idx": -1,
                "kind": cand.get("kind"),
                "anchor": cand.get("anchor"),
                "release_indices": cand.get("release_indices", []),
                "before_score": cur_score,
                "after_score": next_score,
            })
            cur_scene = next_scene
            cur_metrics = next_metrics
            cur_rel = next_rel
            cur_func = next_func
            cur_score = next_score
            role_graph = next_role_graph
            priors = next_priors
            improved = True
            order = _prioritized_object_indices(cur_scene, role_graph, edits, cur_metrics, cfg)

        for idx in order:
            best = _evaluate_best_local_move(
                cur_scene,
                role_graph,
                priors,
                idx,
                edit_by_idx.get(idx),
                cur_score,
                cfg,
                fixed_priors=fixed_priors,
                current_pbl=_get_float_metric(cur_metrics, "total_pbl_loss"),
                timing=timing,
            )
            if best is None:
                continue
            next_scene, next_metrics, next_rel, next_func, next_score, cand, next_role_graph, next_priors = best
            actions.append({
                "round": round_idx,
                "obj_idx": idx,
                "kind": cand.get("kind"),
                "dx": cand.get("dx", 0.0),
                "dz": cand.get("dz", 0.0),
                "dyaw": cand.get("dyaw", 0.0),
                "yaw_abs": cand.get("yaw_abs"),
                "before_score": cur_score,
                "after_score": next_score,
            })
            cur_scene = next_scene
            cur_metrics = next_metrics
            cur_rel = next_rel
            cur_func = next_func
            cur_score = next_score
            role_graph = next_role_graph
            priors = next_priors
            improved = True

        if not improved:
            break

    role_graph = infer_role_graph(cur_scene)
    priors = list(initial_priors or []) if fixed_priors else build_role_based_relation_priors(cur_scene, role_graph)
    cur_pbl_before_post = _get_float_metric(cur_metrics, "total_pbl_loss")
    if not (cfg.skip_post_refine_when_valid_pbl and _is_valid_pbl_value(cur_pbl_before_post, cfg)):
        post_refine_role_layout(
            cur_scene,
            role_graph,
            yaw_from_quaternion,
            quaternion_from_yaw,
            _repair_object_local,
            blend=cfg.role_refine_blend,
            full_repair_passes=cfg.full_repair_after_refine_passes,
        )
    role_graph = infer_role_graph(cur_scene)
    if fixed_priors:
        priors = list(initial_priors or [])
    else:
        det_priors = build_role_based_relation_priors(cur_scene, role_graph)
        priors = _merge_relation_priors(cur_scene, role_graph, det_priors, vlm_priors, cfg.vlm_prior_weight_scale) if vlm_priors and cfg.merge_vlm_with_deterministic else (list(vlm_priors) if vlm_priors else det_priors)
    cur_score, cur_metrics, cur_rel, cur_func = _score_scene_full(cur_scene, role_graph, priors, timing, cfg)
    return cur_scene, cur_metrics, cur_rel, cur_func, actions, priors, role_graph


# ------------------------------
# main loop
# ------------------------------


def optimize_scene_refactored(*, scene: Dict[str, Any], out_root: Path, respace: ReSpace, generator: GPTVLMovePromptGeneratorV5, extra_hints_text: str, cfg: Config) -> Dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    step_runtime_records: List[Dict[str, Any]] = []
    overall_t0 = _now()
    current_scene = _deepcopy_scene(scene)
    timing_stats = TimingStats()
    relation_priors_cache: Optional[List[Dict[str, Any]]] = None
    relation_priors_source = "deterministic"
    history_records: List[Dict[str, Any]] = []
    rejection_memory: Dict[Tuple[str, str, str], int] = {}
    stagnation_count = 0

    initial_role_graph = infer_role_graph(current_scene)
    frozen_deterministic_priors = build_role_based_relation_priors(current_scene, initial_role_graph)
    steps_after_valid_pbl = 0

    # 保存 step_00 开始前的初始分数
    initial_metrics: Optional[Dict[str, Any]] = None
    initial_rel_loss: Optional[float] = None
    initial_func_loss: Optional[float] = None
    initial_score: Optional[float] = None

    for step in range(cfg.max_steps):
        step_t0 = _now()
        step_dir = out_root / f"step_{step:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        _write_json(step_dir / "scene_before.json", current_scene)

        t_render0 = _now()
        respace.render_scene_frame(current_scene, filename=f"step_{step:02d}", pth_viz_output=step_dir)
        diag_path = step_dir / "diag" / f"step_{step:02d}.jpg"
        top_path = render_annotated_top_view(
            current_scene,
            f"step_{step:02d}",
            step_dir,
            resolution=(1024, 1024),
            show_assets=True,
            font_size=14,
        )
        render_sec = _now() - t_render0
        timing_stats.render_sec += render_sec

        role_graph = infer_role_graph(current_scene)
        deterministic_priors = (
            list(frozen_deterministic_priors)
            if cfg.freeze_deterministic_priors
            else build_role_based_relation_priors(current_scene, role_graph)
        )
        planner_mode = "history_replan" if (cfg.use_iteration_history and stagnation_count >= 1) else "normal"
        step_extra_context = _compose_step_extra_context(extra_hints_text, history_records, planner_mode, cfg.max_history_steps)

        if cfg.use_vlm_relation_priors:
            need_refresh = relation_priors_cache is None or cfg.refresh_vlm_relation_priors_every_step or (stagnation_count >= 1 and cfg.refresh_vlm_relation_priors_on_stagnation)
            if need_refresh:
                t_rel0 = _now()
                generated = _safe_generate_relation_priors(
                    generator,
                    diag_path,
                    top_path,
                    current_scene,
                    step_extra_context,
                    cfg.relation_prior_retries,
                    cfg.relation_prior_temperature,
                    cfg.relation_prior_max_tokens,
                    cfg.relation_prior_confidence,
                    step_dir,
                )
                rel_gen_sec = _now() - t_rel0
                timing_stats.vlm_sec += rel_gen_sec
                if generated:
                    relation_priors_cache = generated
                    relation_priors_source = "vlm_relation_priors"
                    _log(f"[step {step:02d}] built {len(relation_priors_cache)} VLM relation priors ({rel_gen_sec:.2f}s)")
                else:
                    relation_priors_cache = None
                    relation_priors_source = "deterministic_fallback"
                    _log(f"[step {step:02d}] VLM relation priors unavailable, fallback to deterministic")

            current_priors = (
                _merge_relation_priors(
                    current_scene,
                    role_graph,
                    deterministic_priors,
                    relation_priors_cache,
                    cfg.vlm_prior_weight_scale,
                )
                if cfg.merge_vlm_with_deterministic
                else list(relation_priors_cache or deterministic_priors)
            )
        else:
            current_priors = deterministic_priors
            relation_priors_cache = None
            relation_priors_source = "deterministic"

        score_before, metrics_before, rel_before, func_before = _score_scene_full(
            current_scene,
            role_graph,
            current_priors,
            timing_stats,
            cfg,
        )
        pbl_before = _get_float_metric(metrics_before, "total_pbl_loss")
        struct_before = metrics_before.get("structure_stats", {}) if isinstance(metrics_before, dict) else {}

        valid_pbl_before = _is_valid_pbl_value(pbl_before, cfg)
        zone_count_before = int(float(struct_before.get("zone_count", 0.0)))
        mono_before = float(struct_before.get("max_zone_ratio", 0.0))
        force_history_replan = cfg.use_iteration_history and (
            stagnation_count >= 1
            or (cfg.force_history_replan_on_monopoly and mono_before > cfg.max_zone_monopoly_ratio)
        )
        planner_mode = "history_replan" if force_history_replan else ("cleanup_only" if (valid_pbl_before and cfg.cleanup_only_after_valid_pbl) else "normal")
        step_extra_context = _compose_step_extra_context(extra_hints_text, history_records, planner_mode, cfg.max_history_steps)
        if force_history_replan and cfg.use_vlm_relation_priors and cfg.refresh_vlm_relation_priors_on_stagnation:
            t_rel0 = _now()
            generated = _safe_generate_relation_priors(
                generator,
                diag_path,
                top_path,
                current_scene,
                step_extra_context,
                cfg.relation_prior_retries,
                cfg.relation_prior_temperature,
                cfg.relation_prior_max_tokens,
                cfg.relation_prior_confidence,
                step_dir / "history_replan_refresh",
            )
            rel_gen_sec = _now() - t_rel0
            timing_stats.vlm_sec += rel_gen_sec
            if generated:
                relation_priors_cache = generated
                relation_priors_source = "vlm_relation_priors_history_refresh"
                current_priors = (
                    _merge_relation_priors(
                        current_scene,
                        role_graph,
                        deterministic_priors,
                        relation_priors_cache,
                        cfg.vlm_prior_weight_scale,
                    )
                    if cfg.merge_vlm_with_deterministic
                    else list(relation_priors_cache or deterministic_priors)
                )
                score_before, metrics_before, rel_before, func_before = _score_scene_full(
                    current_scene,
                    role_graph,
                    current_priors,
                    timing_stats,
                    cfg,
                )
                pbl_before = _get_float_metric(metrics_before, "total_pbl_loss")
                struct_before = metrics_before.get("structure_stats", {}) if isinstance(metrics_before, dict) else {}
                zone_count_before = int(float(struct_before.get("zone_count", 0.0)))
                mono_before = float(struct_before.get("max_zone_ratio", 0.0))

        if step == 0 and initial_score is None:
            initial_metrics = copy.deepcopy(metrics_before)
            initial_rel_loss = float(rel_before)
            initial_func_loss = float(func_before)
            initial_score = float(score_before)

        _write_json(
            step_dir / "role_graph_before.json",
            {
                "categories": role_graph.categories,
                "role_by_idx": role_graph.role_by_idx,
                "function_by_idx": role_graph.function_by_idx,
                "zone_by_idx": role_graph.zone_by_idx,
                "accessory_to_anchor": role_graph.accessory_to_anchor,
                "notes": role_graph.notes,
            },
        )
        _write_json(step_dir / "relation_priors.json", current_priors)
        _write_json(step_dir / "relation_priors_meta.json", {"source": relation_priors_source, "count": len(current_priors)})
        _log(f"[step {step:02d}] relation priors source={relation_priors_source} count={len(current_priors)}")
        struct_before = metrics_before.get("structure_stats", {}) if isinstance(metrics_before, dict) else {}
        _log(
            f"[step {step:02d}] pbl={pbl_before:.6f} oob={_get_float_metric(metrics_before, 'total_oob_loss'):.6f} "
            f"mbl={_get_float_metric(metrics_before, 'total_mbl_loss'):.6f} rel={rel_before:.4f} func={func_before:.4f} "
            f"struct={float(struct_before.get('structure_loss', 0.0)):.4f} open={float(struct_before.get('open_space_ratio', 0.0)):.3f} "
            f"zones={int(float(struct_before.get('zone_count', 0.0)))} mono={float(struct_before.get('max_zone_ratio', 0.0)):.3f} "
            f"score={score_before:.6f} valid_pbl={valid_pbl_before}"
        )

        if step > 0 and cfg.stop_when_valid_pbl and valid_pbl_before and score_before <= cfg.stop_score_threshold:
            _log(
                f"[step {step:02d}] early stop: score already good enough "
                f"(score={score_before:.6f} <= {cfg.stop_score_threshold:.6f})"
            )
            break

        if valid_pbl_before and cfg.cleanup_only_after_valid_pbl and not force_history_replan:
            prompt_sec = 0.0
            parse_sec = 0.0
            applied_count = 0
            changes: List[Dict[str, Any]] = []
            parse_result = parse_move_prompt("")
            scene_after_prompt = _deepcopy_scene(current_scene)
            _write_text(step_dir / "move_prompt.txt", "[cleanup-only-after-valid-pbl]")
            _write_json(
                step_dir / "move_prompt_parse.json",
                {
                    "room_name": "",
                    "header_line": "",
                    "parse_warnings": ["cleanup_only_after_valid_pbl"],
                    "num_edits": 0,
                    "edits": [],
                },
            )
            _write_json(step_dir / "applied_edits.json", {"applied_count": 0, "changes": []})

            work_cfg = copy.deepcopy(cfg)
            work_cfg.max_rounds = min(cfg.max_rounds_after_valid_pbl, cfg.max_rounds)
            work_cfg.max_objects_per_round = min(cfg.max_objects_after_valid_pbl, cfg.max_objects_per_round)
            work_cfg.step_xy = min(cfg.step_xy, 0.10)
            work_cfg.step_yaw = min(cfg.step_yaw, 5.0)
            work_cfg.proxy_topk = 3

            t_opt0 = _now()
            optimized_scene, metrics_after, rel_after, func_after, actions, priors_after, role_graph_after = _optimize_after_prompt(
                scene_after_prompt,
                [],
                work_cfg,
                initial_priors=current_priors,
                fixed_priors=False,
                vlm_priors=relation_priors_cache if cfg.use_vlm_relation_priors and relation_priors_source.startswith("vlm") else None,
                timing=timing_stats,
            )
        else:
            t_prompt0 = _now()
            prompt_result = _safe_generate_move_prompt(
                generator,
                diag_path,
                top_path,
                current_scene,
                step_extra_context,
                cfg.move_prompt_retries,
                cfg.move_prompt_temperature,
                cfg.move_prompt_max_tokens,
                step_dir,
            )
            prompt_sec = _now() - t_prompt0
            timing_stats.vlm_sec += prompt_sec
            _write_text(step_dir / "move_prompt.txt", prompt_result.move_prompt)
            _log(f"[step {step:02d}] move_prompt generated ({len(prompt_result.move_prompt)} chars), temp={cfg.move_prompt_temperature}")

            t_parse0 = _now()
            parse_result = parse_move_prompt(prompt_result.move_prompt)
            parse_sec = _now() - t_parse0
            _write_json(
                step_dir / "move_prompt_parse.json",
                {
                    "room_name": parse_result.room_name,
                    "header_line": parse_result.header_line,
                    "parse_warnings": parse_result.parse_warnings,
                    "num_edits": len(parse_result.edits),
                    "edits": [e.__dict__ for e in parse_result.edits],
                },
            )

            scene_after_prompt = _deepcopy_scene(current_scene)
            scene_after_prompt, applied_count, changes = apply_edits_to_scene(scene_after_prompt, parse_result.edits)
            _write_json(step_dir / "applied_edits.json", {"applied_count": applied_count, "changes": changes})

            t_opt0 = _now()
            optimized_scene, metrics_after, rel_after, func_after, actions, priors_after, role_graph_after = _optimize_after_prompt(
                scene_after_prompt,
                parse_result.edits,
                cfg,
                initial_priors=current_priors,
                fixed_priors=False,
                vlm_priors=relation_priors_cache if cfg.use_vlm_relation_priors and relation_priors_source.startswith("vlm") else None,
                timing=timing_stats,
            )

        opt_sec = _now() - t_opt0
        timing_stats.optimize_sec += opt_sec
        struct_after = metrics_after.get("structure_stats", {}) if isinstance(metrics_after, dict) else {}
        score_after = (
            _W_PBL * _get_float_metric(metrics_after, "total_pbl_loss")
            + _W_REL * rel_after
            + _W_FUNC * func_after
            + float(struct_after.get("structure_loss", 0.0))
        )

        t_post0 = _now()
        _write_json(step_dir / "optimizer_actions.json", actions)
        _write_json(step_dir / "scene_after.json", optimized_scene)
        _write_json(
            step_dir / "role_graph_after.json",
            {
                "categories": role_graph_after.categories,
                "role_by_idx": role_graph_after.role_by_idx,
                "function_by_idx": role_graph_after.function_by_idx,
                "zone_by_idx": role_graph_after.zone_by_idx,
                "accessory_to_anchor": role_graph_after.accessory_to_anchor,
                "notes": role_graph_after.notes,
            },
        )
        _write_json(step_dir / "relation_priors_after.json", priors_after)
        _write_json(step_dir / "relation_priors_after_meta.json", {"source": relation_priors_source, "count": len(priors_after)})
        post_sec = _now() - t_post0

        total_sec = _now() - step_t0
        _log(
            f"[step {step:02d}] time total={total_sec:.2f}s (prompt={prompt_sec:.2f}s, parse={parse_sec:.2f}s, "
            f"opt={opt_sec:.2f}s, post={post_sec:.2f}s) applied={applied_count} "
            f"pbl {pbl_before:.6f}->{_get_float_metric(metrics_after, 'total_pbl_loss'):.6f} "
            f"rel {rel_before:.4f}->{rel_after:.4f} "
            f"func {func_before:.4f}->{func_after:.4f} "
            f"struct {float(struct_before.get('structure_loss', 0.0)):.4f}->{float(struct_after.get('structure_loss', 0.0)):.4f} "
            f"open {float(struct_before.get('open_space_ratio', 0.0)):.3f}->{float(struct_after.get('open_space_ratio', 0.0)):.3f} "
            f"zones {int(float(struct_before.get('zone_count', 0.0)))}->{int(float(struct_after.get('zone_count', 0.0)))} "
            f"mono {float(struct_before.get('max_zone_ratio', 0.0)):.3f}->{float(struct_after.get('max_zone_ratio', 0.0)):.3f} "
            f"score {score_before:.6f}->{score_after:.6f}"
        )

        pbl_after = _get_float_metric(metrics_after, "total_pbl_loss")
        accepted = False
        reject_reason = "no_improvement"
        before_hash = _scene_state_hash(current_scene)
        after_hash = _scene_state_hash(optimized_scene)
        zone_count_after = int(float(struct_after.get("zone_count", 0.0)))
        open_before = float(struct_before.get("open_space_ratio", 0.0))
        open_after = float(struct_after.get("open_space_ratio", 0.0))
        mono_before = float(struct_before.get("max_zone_ratio", 0.0))
        mono_after = float(struct_after.get("max_zone_ratio", 0.0))
        spread_before = float(struct_before.get("spread_penalty", 0.0))
        spread_after = float(struct_after.get("spread_penalty", 0.0))
        flow_before = float(struct_before.get("flow_penalty", 0.0))
        flow_after = float(struct_after.get("flow_penalty", 0.0))

        if valid_pbl_before:
            if not _is_valid_pbl_value(pbl_after, cfg):
                reject_reason = "reintroduced_pbl"
            elif cfg.use_structural_guard and cfg.require_zone_count_preserve_after_valid_pbl and zone_count_before >= 2 and zone_count_after < zone_count_before:
                reject_reason = "zone_count_drop"
            elif cfg.use_structural_guard and open_after < open_before - cfg.max_open_space_drop_after_valid_pbl:
                reject_reason = "open_space_drop"
            elif cfg.use_structural_guard and mono_after > mono_before + cfg.max_monopoly_increase_after_valid_pbl:
                reject_reason = "zone_monopoly_worse"
            elif cfg.use_structural_guard and spread_after > spread_before + cfg.max_spread_increase_after_valid_pbl:
                reject_reason = "spread_worse"
            elif cfg.use_structural_guard and flow_after > flow_before + cfg.max_flow_increase_after_valid_pbl:
                reject_reason = "flow_worse"
            elif rel_after > rel_before + cfg.max_rel_increase_after_valid_pbl:
                if cfg.adaptive_relation_tradeoff and _history_relayout_worthy(score_before, score_after, struct_before, struct_after, cfg):
                    rewrite_score, rewrite_metrics, rewrite_rel, rewrite_func, rewrite_priors = _rescore_with_counterfactual_priors(
                        optimized_scene,
                        role_graph_after,
                        timing_stats,
                        cfg,
                    )
                    _write_json(step_dir / "counterfactual_rewrite_priors.json", rewrite_priors)
                    _write_json(step_dir / "counterfactual_rewrite_metrics.json", rewrite_metrics)
                    _log(
                        f"[step {step:02d}] counterfactual prior rewrite rel {rel_after:.4f}->{rewrite_rel:.4f} "
                        f"score {score_after:.6f}->{rewrite_score:.6f}"
                    )
                    if rewrite_score < score_before - cfg.min_score_improve_after_valid_pbl:
                        score_after = rewrite_score
                        metrics_after = rewrite_metrics
                        rel_after = rewrite_rel
                        func_after = rewrite_func
                        priors_after = rewrite_priors
                        struct_after = metrics_after.get("structure_stats", {}) if isinstance(metrics_after, dict) else {}
                        zone_count_after = int(float(struct_after.get("zone_count", 0.0)))
                        open_after = float(struct_after.get("open_space_ratio", 0.0))
                        mono_after = float(struct_after.get("max_zone_ratio", 0.0))
                        spread_after = float(struct_after.get("spread_penalty", 0.0))
                        flow_after = float(struct_after.get("flow_penalty", 0.0))
                        accepted = True
                        relation_priors_source = "counterfactual_deterministic_rewrite"
                    else:
                        reject_reason = "relation_worse"
                else:
                    reject_reason = "relation_worse"
            elif func_after > func_before + cfg.max_func_increase_after_valid_pbl:
                reject_reason = "functional_worse"
            elif score_after < score_before - cfg.min_score_improve_after_valid_pbl:
                accepted = True
            else:
                reject_reason = "improvement_too_small"
        else:
            pbl_improved = pbl_after < pbl_before - cfg.monotonic_eps
            score_guard = score_after <= score_before + cfg.max_score_increase_prevalid
            rel_guard = rel_after <= rel_before + cfg.max_rel_increase_prevalid
            func_guard = func_after <= func_before + cfg.max_func_increase_prevalid
            struct_guard = True
            if cfg.use_structural_guard:
                struct_guard = (
                    zone_count_after >= max(1, zone_count_before - 1)
                    and open_after >= open_before - max(0.08, cfg.max_open_space_drop_after_valid_pbl)
                    and mono_after <= max(mono_before + 0.10, cfg.max_zone_monopoly_ratio + 0.05)
                )

            if pbl_improved and score_guard and rel_guard and func_guard and struct_guard:
                accepted = True
            elif score_after < score_before - cfg.monotonic_eps and pbl_after <= pbl_before + 0.005 and func_guard and struct_guard:
                accepted = True
            else:
                if not pbl_improved:
                    reject_reason = "pbl_not_better"
                elif not struct_guard:
                    reject_reason = "structure_worse_too_much"
                elif not score_guard:
                    reject_reason = "score_worse_too_much"
                elif not rel_guard:
                    reject_reason = "relation_worse_too_much"
                else:
                    reject_reason = "functional_worse_too_much"

        diagnosis = ""
        if not accepted:
            if reject_reason == "relation_worse":
                diagnosis = "Previous candidate improved structure but broke current relation priors; next step should propose a relation-consistent re-layout or a smaller bridge edit."
            elif str(reject_reason).startswith("repeat_"):
                diagnosis = "A near-duplicate candidate has already been rejected; next step must diversify the edit target or edit order."
            elif reject_reason in {"open_space_drop", "zone_monopoly_worse", "spread_worse", "flow_worse"}:
                diagnosis = "The candidate harmed global structure; next step should preserve circulation and secondary zones."
        history_item = {
            "step": step,
            "planner_mode": planner_mode,
            "accepted": accepted,
            "reject_reason": None if accepted else reject_reason,
            "score_before": float(score_before),
            "score_after": float(score_after),
            "rel_before": float(rel_before),
            "rel_after": float(rel_after),
            "func_before": float(func_before),
            "func_after": float(func_after),
            "struct_before": float(struct_before.get("structure_loss", 0.0)),
            "struct_after": float(struct_after.get("structure_loss", 0.0)),
            "zone_before": zone_count_before,
            "zone_after": zone_count_after,
            "mono_before": mono_before,
            "mono_after": mono_after,
            "diagnosis": diagnosis,
            "before_hash": before_hash,
            "after_hash": after_hash,
        }
        history_records.append(history_item)
        if len(history_records) > max(2, cfg.max_history_steps * 2):
            history_records = history_records[-max(2, cfg.max_history_steps * 2):]

        if accepted:
            current_scene = optimized_scene
            stagnation_count = 0
            if cfg.use_vlm_relation_priors and (cfg.refresh_vlm_relation_priors_every_step or cfg.refresh_vlm_relation_priors_on_stagnation):
                relation_priors_cache = None
            if _is_valid_pbl_value(pbl_after, cfg):
                steps_after_valid_pbl = steps_after_valid_pbl + 1 if valid_pbl_before else 0
            else:
                steps_after_valid_pbl = 0
            _log(f"[step {step:02d}] accepted")
        else:
            rej_key = (before_hash, after_hash, str(reject_reason))
            rejection_memory[rej_key] = rejection_memory.get(rej_key, 0) + 1
            if rejection_memory[rej_key] >= cfg.repeat_reject_patience:
                reject_reason = f"repeat_{reject_reason}"
                history_records[-1]["reject_reason"] = reject_reason
                history_records[-1]["diagnosis"] = "Repeated rejected proposal detected. Force VLM history-guided replanning and refresh relation priors next step."
                stagnation_count += 1
                if cfg.refresh_vlm_relation_priors_on_stagnation:
                    relation_priors_cache = None
            else:
                stagnation_count = max(1, stagnation_count)
            _log(f"[step {step:02d}] rejected: {reject_reason}")

        step_runtime_records.append(
            {
                "step": step,
                "runtime_sec": round(total_sec, 4),
                "accepted": accepted,
                "relation_priors_source": relation_priors_source,
                "num_relation_priors": len(priors_after),
                "render_sec": round(render_sec, 4),
                "prompt_sec": round(prompt_sec, 4),
                "opt_sec": round(opt_sec, 4),
                "eval_sec_accum": round(timing_stats.eval_sec, 4),
                "open_space_ratio_after": round(open_after, 4),
                "zone_count_after": zone_count_after,
                "max_zone_ratio_after": round(mono_after, 4),
                "structure_loss_after": round(float(struct_after.get("structure_loss", 0.0)), 4),
                "planner_mode": planner_mode,
                "reject_reason": None if accepted else reject_reason,
                "stagnation_count": stagnation_count,
            }
        )

        if (
            cfg.stop_when_valid_pbl
            and accepted
            and _is_valid_pbl_value(_get_float_metric(metrics_after, "total_pbl_loss"), cfg)
            and score_after <= cfg.stop_score_threshold
        ):
            _log(
                f"[step {step:02d}] early stop: accepted score is good enough "
                f"(score={score_after:.6f} <= {cfg.stop_score_threshold:.6f})"
            )
            break

    if cfg.render_final:
        final_dir = out_root / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        t_render0 = _now()
        respace.render_scene_frame(current_scene, filename="final", pth_viz_output=final_dir)
        render_annotated_top_view(
            current_scene,
            "final",
            final_dir,
            resolution=(1024, 1024),
            show_assets=True,
            font_size=14,
        )
        timing_stats.render_sec += _now() - t_render0
        _write_json(final_dir / "scene.json", current_scene)

    final_role_graph = infer_role_graph(current_scene)
    deterministic_priors = (
        list(frozen_deterministic_priors)
        if cfg.freeze_deterministic_priors
        else build_role_based_relation_priors(current_scene, final_role_graph)
    )
    if cfg.use_vlm_relation_priors and relation_priors_cache:
        priors = (
            _merge_relation_priors(
                current_scene,
                final_role_graph,
                deterministic_priors,
                relation_priors_cache,
                cfg.vlm_prior_weight_scale,
            )
            if cfg.merge_vlm_with_deterministic
            else relation_priors_cache
        )
    else:
        priors = deterministic_priors
        if not cfg.use_vlm_relation_priors:
            relation_priors_source = "deterministic"

    final_score, final_metrics, final_rel_loss, final_func_loss = _score_scene_full(
        current_scene,
        final_role_graph,
        priors,
        timing_stats,
        cfg,
    )

    summary = {
        "relation_priors_source": relation_priors_source,
        "num_relation_priors": len(priors),
        "total_runtime_sec": round(_now() - overall_t0, 4),
        "render_sec": round(timing_stats.render_sec, 4),
        "vlm_sec": round(timing_stats.vlm_sec, 4),
        "optimize_sec": round(timing_stats.optimize_sec, 4),
        "eval_sec": round(timing_stats.eval_sec, 4),
        "step_runtime_records": step_runtime_records,
        "history_records": history_records,

        "initial_metrics": initial_metrics,
        "initial_rel_loss": None if initial_rel_loss is None else round(initial_rel_loss, 4),
        "initial_func_loss": None if initial_func_loss is None else round(initial_func_loss, 4),
        "initial_structure_stats": None if not isinstance(initial_metrics, dict) else initial_metrics.get("structure_stats"),
        "initial_score": None if initial_score is None else round(initial_score, 6),

        "final_metrics": final_metrics,
        "final_rel_loss": round(final_rel_loss, 4),
        "final_func_loss": round(final_func_loss, 4),
        "final_structure_stats": final_metrics.get("structure_stats") if isinstance(final_metrics, dict) else None,
        "final_score": round(final_score, 6),
    }

    _write_json(out_root / "summary.json", summary)
    return summary


# ------------------------------
# CLI
# ------------------------------

def main() -> None:
    if os.getenv("YUNWU_AI_API_BASE") is None and os.getenv("YUNWU_AI_BASE_URL") is not None:
        os.environ["YUNWU_AI_API_BASE"] = os.environ["YUNWU_AI_BASE_URL"]
    if not os.getenv("YUNWU_AI_API_KEY"):
        raise RuntimeError("Missing YUNWU_AI_API_KEY env var.")

    respace = ReSpace()
    scene_json_path = Path(os.getenv("SCENE_JSON_PATH", "")).expanduser()
    if not scene_json_path.exists():
        raise FileNotFoundError(f"scene json not found: {scene_json_path}")
    scene = json.loads(scene_json_path.read_text(encoding="utf-8"))

    room_prompt = os.getenv("ROOM_PROMPT", "").strip()
    updated_scene, is_success = respace.handle_prompt(room_prompt, scene)
    _ = is_success

    out_root = Path(os.getenv("OUT_DIR", "./evaluate/gpt_image_describe_role_refactor")).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        max_steps=int(os.getenv("MAX_STEPS", "3")),
        max_rounds=int(os.getenv("MAX_ROUNDS", "2")),
        max_objects_per_round=int(os.getenv("MAX_OBJECTS_PER_ROUND", "8")),
        proxy_topk=int(os.getenv("PROXY_TOPK", "3")),
        move_prompt_temperature=float(os.getenv("MOVE_PROMPT_TEMPERATURE", "0.2")),
        move_prompt_max_tokens=int(os.getenv("MOVE_PROMPT_MAX_TOKENS", "1200")),
        move_prompt_retries=int(os.getenv("MOVE_PROMPT_RETRIES", "2")),
        relation_prior_retries=int(os.getenv("RELATION_PRIOR_RETRIES", "2")),
        relation_prior_temperature=float(os.getenv("RELATION_PRIOR_TEMPERATURE", "0.0")),
        relation_prior_max_tokens=int(os.getenv("RELATION_PRIOR_MAX_TOKENS", "900")),
        relation_prior_confidence=float(os.getenv("RELATION_PRIOR_CONFIDENCE", "0.55")),
        use_vlm_relation_priors=os.getenv("USE_VLM_RELATION_PRIORS", "1") not in ("0", "false", "False", ""),
        refresh_vlm_relation_priors_every_step=os.getenv("REFRESH_VLM_RELATION_PRIORS_EVERY_STEP", "0") not in ("0", "false", "False", ""),
        merge_vlm_with_deterministic=os.getenv("MERGE_VLM_WITH_DETERMINISTIC", "1") not in ("0", "false", "False", ""),
        vlm_prior_weight_scale=float(os.getenv("VLM_PRIOR_WEIGHT_SCALE", "0.35")),
        freeze_deterministic_priors=os.getenv("FREEZE_DETERMINISTIC_PRIORS", "0") not in ("0", "false", "False", ""),
        local_repair_passes=int(os.getenv("LOCAL_REPAIR_PASSES", "1")),
        full_repair_after_refine_passes=int(os.getenv("FULL_REPAIR_AFTER_REFINE_PASSES", "1")),
        stop_when_valid_pbl=os.getenv("STOP_WHEN_VALID_PBL", "1") not in ("0", "false", "False", ""),
        cleanup_only_after_valid_pbl=os.getenv("CLEANUP_ONLY_AFTER_VALID_PBL", "1") not in ("0", "false", "False", ""),
        max_steps_after_valid_pbl=int(os.getenv("MAX_STEPS_AFTER_VALID_PBL", "1")),
        max_objects_after_valid_pbl=int(os.getenv("MAX_OBJECTS_AFTER_VALID_PBL", "4")),
        max_rounds_after_valid_pbl=int(os.getenv("MAX_ROUNDS_AFTER_VALID_PBL", "1")),
        min_score_improve_after_valid_pbl=float(os.getenv("MIN_SCORE_IMPROVE_AFTER_VALID_PBL", "0.02")),
        max_rel_increase_after_valid_pbl=float(os.getenv("MAX_REL_INCREASE_AFTER_VALID_PBL", "0.12")),
        max_func_increase_after_valid_pbl=float(os.getenv("MAX_FUNC_INCREASE_AFTER_VALID_PBL", "0.12")),
        max_score_increase_prevalid=float(os.getenv("MAX_SCORE_INCREASE_PREVALID", "0.06")),
        max_rel_increase_prevalid=float(os.getenv("MAX_REL_INCREASE_PREVALID", "0.55")),
        max_func_increase_prevalid=float(os.getenv("MAX_FUNC_INCREASE_PREVALID", "0.35")),
        render_final=os.getenv("RENDER_FINAL", "1") not in ("0", "false", "False", ""),
        monotonic_eps=float(os.getenv("MONOTONIC_EPS", "1e-12")),
        step_xy=float(os.getenv("STEP_XY", "0.22")),
        step_yaw=float(os.getenv("STEP_YAW", "15.0")),
        anchor_lock_pbl_threshold=float(os.getenv("ANCHOR_LOCK_PBL_THRESHOLD", "0.08")),
        role_refine_blend=float(os.getenv("ROLE_REFINE_BLEND", "0.66")),
        valid_pbl_threshold=float(os.getenv("VALID_PBL_THRESHOLD", "0.10")),
        stop_score_threshold=float(os.getenv("STOP_SCORE_THRESHOLD", "0.80")),
        skip_post_refine_when_valid_pbl=os.getenv("SKIP_POST_REFINE_WHEN_VALID_PBL", "1") not in ("0", "false", "False", ""),
        candidate_filter_reintroduced_pbl=os.getenv("CANDIDATE_FILTER_REINTRODUCED_PBL", "1") not in ("0", "false", "False", ""),
        use_structural_guard=os.getenv("USE_STRUCTURAL_GUARD", "1") not in ("0", "false", "False", ""),
        min_open_space_ratio=float(os.getenv("MIN_OPEN_SPACE_RATIO", "0.42")),
        max_zone_monopoly_ratio=float(os.getenv("MAX_ZONE_MONOPOLY_RATIO", "0.72")),
        corridor_width_ratio=float(os.getenv("CORRIDOR_WIDTH_RATIO", "0.18")),
        max_open_space_drop_after_valid_pbl=float(os.getenv("MAX_OPEN_SPACE_DROP_AFTER_VALID_PBL", "0.06")),
        max_monopoly_increase_after_valid_pbl=float(os.getenv("MAX_MONOPOLY_INCREASE_AFTER_VALID_PBL", "0.08")),
        max_spread_increase_after_valid_pbl=float(os.getenv("MAX_SPREAD_INCREASE_AFTER_VALID_PBL", "0.10")),
        max_flow_increase_after_valid_pbl=float(os.getenv("MAX_FLOW_INCREASE_AFTER_VALID_PBL", "0.10")),
        require_zone_count_preserve_after_valid_pbl=os.getenv("REQUIRE_ZONE_COUNT_PRESERVE_AFTER_VALID_PBL", "1") not in ("0", "false", "False", ""),
        add_zone_release_candidates=os.getenv("ADD_ZONE_RELEASE_CANDIDATES", "1") not in ("0", "false", "False", ""),
        zone_release_inset_ratio=float(os.getenv("ZONE_RELEASE_INSET_RATIO", "0.16")),
        enable_zone_layout_candidates=os.getenv("ENABLE_ZONE_LAYOUT_CANDIDATES", "1") not in ("0", "false", "False", ""),
        zone_layout_trigger_ratio=float(os.getenv("ZONE_LAYOUT_TRIGGER_RATIO", "0.76")),
        max_zone_layout_candidates=int(os.getenv("MAX_ZONE_LAYOUT_CANDIDATES", "4")),
        zone_layout_topk_objects=int(os.getenv("ZONE_LAYOUT_TOPK_OBJECTS", "3")),
        zone_layout_min_score_improve=float(os.getenv("ZONE_LAYOUT_MIN_SCORE_IMPROVE", "0.01")),
        zone_layout_anchor_inset_ratio=float(os.getenv("ZONE_LAYOUT_ANCHOR_INSET_RATIO", "0.14")),
        zone_layout_secondary_cluster_radius=float(os.getenv("ZONE_LAYOUT_SECONDARY_CLUSTER_RADIUS", "0.55")),
        zone_layout_pair_table_chair=os.getenv("ZONE_LAYOUT_PAIR_TABLE_CHAIR", "1") not in ("0", "false", "False", ""),
        use_iteration_history=os.getenv("USE_ITERATION_HISTORY", "1") not in ("0", "false", "False", ""),
        max_history_steps=int(os.getenv("MAX_HISTORY_STEPS", "4")),
        repeat_reject_patience=int(os.getenv("REPEAT_REJECT_PATIENCE", "2")),
        adaptive_relation_tradeoff=os.getenv("ADAPTIVE_RELATION_TRADEOFF", "1") not in ("0", "false", "False", ""),
        rewrite_priors_on_relayout=os.getenv("REWRITE_PRIORS_ON_RELAYOUT", "1") not in ("0", "false", "False", ""),
        refresh_vlm_relation_priors_on_stagnation=os.getenv("REFRESH_VLM_RELATION_PRIORS_ON_STAGNATION", "1") not in ("0", "false", "False", ""),
        force_history_replan_on_monopoly=os.getenv("FORCE_HISTORY_REPLAN_ON_MONOPOLY", "1") not in ("0", "false", "False", ""),
        relayout_accept_min_score_gain=float(os.getenv("RELAYOUT_ACCEPT_MIN_SCORE_GAIN", "0.03")),
        relayout_accept_min_structure_gain=float(os.getenv("RELAYOUT_ACCEPT_MIN_STRUCTURE_GAIN", "0.01")),
        relayout_accept_min_monopoly_gain=float(os.getenv("RELAYOUT_ACCEPT_MIN_MONOPOLY_GAIN", "0.15")),
        relayout_accept_min_zone_gain=int(os.getenv("RELAYOUT_ACCEPT_MIN_ZONE_GAIN", "1")),
    )

    extra_hints_text = (
        "GLOBAL SAFETY CONSTRAINTS:\n"
        "1) All objects must stay fully inside the room.\n"
        "2) Avoid overlaps; keep small but visible clearance.\n"
        "3) Prioritize OOB and collision fixes before aesthetics.\n"
        "4) Preserve dominant-anchor and accessory structure.\n"
        "5) Keep interactive fronts usable and avoid over-crowding one functional anchor.\n"
        "6) Preserve secondary functional zones; do not collapse all seating into one dominant cluster.\n"
        "7) Maintain visible open space and a clear main circulation band through the room.\n"
        "8) Use the step history to avoid repeating rejected edits; propose a different diagnose-and-act plan when the last attempt failed.\n"
    )

    model = os.getenv("MOVE_PROMPT_MODEL", os.getenv("YUNWU_AI_MODEL", "gpt-4o"))
    generator = GPTVLMovePromptGeneratorV5(model=model, api_base=os.getenv("YUNWU_AI_API_BASE"), api_key=os.getenv("YUNWU_AI_API_KEY"), timeout_s=float(os.getenv("MOVE_PROMPT_TIMEOUT_S", "120")))

    summary = optimize_scene_refactored(scene=updated_scene, out_root=out_root, respace=respace, generator=generator, extra_hints_text=extra_hints_text, cfg=cfg)
    _log("\n=== Done ===")
    _log(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
