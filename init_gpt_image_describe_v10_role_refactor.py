from __future__ import annotations

import copy
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
    max_dir_increase_after_valid_pbl: float = 0.04
    max_rel_increase_after_valid_pbl: float = 0.12
    max_func_increase_after_valid_pbl: float = 0.12
    max_score_increase_prevalid: float = 0.06
    max_dir_increase_prevalid: float = 0.12
    max_rel_increase_prevalid: float = 0.55
    max_func_increase_prevalid: float = 0.35
    render_final: bool = True
    monotonic_eps: float = 1e-12
    step_xy: float = 0.22
    step_yaw: float = 15.0
    anchor_lock_pbl_threshold: float = 0.08
    role_refine_blend: float = 0.66


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
# direction + relation + functional losses
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


def _target_yaw_for_object(scene: Dict[str, Any], role_graph: RoleGraph, idx: int) -> Optional[float]:
    obj = scene["objects"][idx]
    cat = role_graph.categories[idx]
    pos = obj.get("pos", [0.0, 0.0, 0.0])
    yaw = yaw_from_quaternion(obj.get("rot", [0.0, 0.0, 0.0, 1.0]))

    if idx in role_graph.accessory_to_anchor:
        anchor_idx = role_graph.accessory_to_anchor[idx]
        _, target_yaw = target_pose_for_attachment(scene, scene["objects"], idx, anchor_idx, yaw_from_quaternion)
        return target_yaw

    if cat == "bed":
        return _nearest_normal_axis_yaw(scene, pos, yaw)
    if idx in role_graph.parallel_wall_indices:
        return _nearest_parallel_wall_yaw(scene, pos, yaw)
    if idx in role_graph.wall_affine_indices:
        wall_yaw = _find_nearest_wall_yaw(scene, pos)
        if wall_yaw is not None:
            return _normalize_angle(wall_yaw + 180.0)
    if cat in {"sofa", "tv stand"}:
        cx, cz = room_center_xz(scene)
        return _normalize_angle(math.degrees(math.atan2(cx - pos[0], cz - pos[2])))
    return None


def _compute_direction_loss(scene: Dict[str, Any], role_graph: RoleGraph) -> Tuple[float, List[Dict[str, Any]]]:
    violations: List[Dict[str, Any]] = []
    total = 0.0
    for i, obj in enumerate(scene.get("objects", [])):
        target_yaw = _target_yaw_for_object(scene, role_graph, i)
        if target_yaw is None:
            continue
        current_yaw = yaw_from_quaternion(obj.get("rot", [0.0, 0.0, 0.0, 1.0]))
        diff = _angle_diff(current_yaw, target_yaw)
        threshold = 10.0 if role_graph.categories[i] == "bed" else 22.0 if i in role_graph.accessory_to_anchor else 28.0
        weight = 1.0 if role_graph.categories[i] in {"bed", "desk", "table", "counter", "sink", "toilet"} else 0.7
        if diff > threshold:
            penalty = diff / 180.0 * weight
            total += penalty
            violations.append({"idx": i, "cat": role_graph.categories[i], "current_yaw": round(current_yaw, 1), "target_yaw": round(target_yaw, 1), "penalty": round(penalty, 4)})
    return total, violations


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
_W_DIR = 0.10
_W_REL = 0.15
_W_FUNC = 0.25


def _get_float_metric(metrics: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else default


def _score_scene_full(
    scene: Dict[str, Any],
    role_graph: Optional[RoleGraph] = None,
    relation_priors: Optional[List[Dict[str, Any]]] = None,
    timing_stats: Optional[TimingStats] = None,
) -> Tuple[float, Dict[str, Any], float, float, float]:
    t0 = time.perf_counter()
    metrics = eval_scene(scene, is_debug=False)
    if role_graph is None:
        role_graph = infer_role_graph(scene)
    priors = relation_priors or []
    dir_loss, _ = _compute_direction_loss(scene, role_graph)
    rel_loss, _ = _compute_relation_loss(scene, role_graph, priors)
    func_loss = compute_functional_loss(scene, role_graph, yaw_from_quaternion).total
    score = (
        _W_PBL * _get_float_metric(metrics, "total_pbl_loss")
        + _W_DIR * dir_loss
        + _W_REL * rel_loss
        + _W_FUNC * func_loss
    )
    if timing_stats is not None:
        timing_stats.eval_sec += time.perf_counter() - t0
    return score, metrics, dir_loss, rel_loss, func_loss


def _collision_area_for_object(scene: Dict[str, Any], idx: int) -> float:
    return sum(area for _, _, _, area in _collision_neighbors(scene, idx))


def _direction_penalty_for_object(scene: Dict[str, Any], role_graph: RoleGraph, idx: int) -> float:
    _, violations = _compute_direction_loss(scene, role_graph)
    return sum(v["penalty"] for v in violations if v.get("idx") == idx)


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


def _quick_candidate_proxy_score(scene: Dict[str, Any], role_graph: RoleGraph, priors: List[Dict[str, Any]], idx: int) -> float:
    floor_polygon = _get_floor_polygon(scene)
    oob = 0.0
    if floor_polygon is not None:
        oob = compute_oob(scene["objects"][idx], floor_polygon, scene.get("bounds_bottom", []), scene.get("bounds_top", []), is_debug=False)
    collision = _collision_area_for_object(scene, idx)
    direction = _direction_penalty_for_object(scene, role_graph, idx)
    relation = _relation_penalty_for_object(scene, role_graph, priors, idx)
    functional = _functional_penalty_for_object(scene, role_graph, idx)
    return 3.0 * oob + 2.0 * collision + 1.0 * direction + 1.2 * relation + 1.4 * functional


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
    return optimization_stage_order(scene, role_graph, edited_indices, hotspot_idx, cfg.max_objects_per_round, lock_major)


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
    timing: Optional[TimingStats] = None,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], float, float, float, float, Dict[str, Any], RoleGraph, List[Dict[str, Any]]]]:
    candidates = _generate_candidates(scene, role_graph, idx, bias_edit, cfg)
    proxy_ranked: List[Tuple[float, Dict[str, Any], Dict[str, Any], RoleGraph]] = []

    for cand in candidates:
        sc = _apply_delta(scene, idx, dx=float(cand.get("dx", 0.0)), dz=float(cand.get("dz", 0.0)), dyaw=float(cand.get("dyaw", 0.0)), yaw_abs=None if cand.get("yaw_abs") is None else float(cand.get("yaw_abs")))
        _repair_object_local(sc, idx, cfg.local_repair_passes)
        local_role_graph = infer_role_graph(sc)
        local_priors = priors if fixed_priors else build_role_based_relation_priors(sc, local_role_graph)
        proxy = _quick_candidate_proxy_score(sc, local_role_graph, local_priors, idx)
        proxy_ranked.append((proxy, cand, sc, local_role_graph))

    proxy_ranked.sort(key=lambda x: x[0])
    best = None
    for _, cand, sc, local_role_graph in proxy_ranked[: max(1, cfg.proxy_topk)]:
        local_priors = priors if fixed_priors else build_role_based_relation_priors(sc, local_role_graph)
        score, metrics, dir_loss, rel_loss, func_loss = _score_scene_full(sc, local_role_graph, local_priors, timing)
        if score < current_score - cfg.monotonic_eps:
            if best is None or score < best[5]:
                best = (sc, metrics, dir_loss, rel_loss, func_loss, score, cand, local_role_graph, local_priors)
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
) -> Tuple[Dict[str, Any], Dict[str, Any], float, float, float, List[Dict[str, Any]], List[Dict[str, Any]], RoleGraph]:
    cur_scene = _deepcopy_scene(scene)
    role_graph = infer_role_graph(cur_scene)
    if fixed_priors:
        priors = list(initial_priors or [])
    else:
        det_priors = build_role_based_relation_priors(cur_scene, role_graph)
        priors = _merge_relation_priors(cur_scene, role_graph, det_priors, vlm_priors, cfg.vlm_prior_weight_scale) if vlm_priors and cfg.merge_vlm_with_deterministic else (list(vlm_priors) if vlm_priors else det_priors)
    cur_score, cur_metrics, cur_dir, cur_rel, cur_func = _score_scene_full(cur_scene, role_graph, priors, timing)
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

        for idx in order:
            best = _evaluate_best_local_move(cur_scene, role_graph, priors, idx, edit_by_idx.get(idx), cur_score, cfg, fixed_priors=fixed_priors, timing=timing)
            if best is None:
                continue
            next_scene, next_metrics, next_dir, next_rel, next_func, next_score, cand, next_role_graph, next_priors = best
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
            cur_dir = next_dir
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
    post_refine_role_layout(cur_scene, role_graph, yaw_from_quaternion, quaternion_from_yaw, _repair_object_local, blend=cfg.role_refine_blend, full_repair_passes=cfg.full_repair_after_refine_passes)
    role_graph = infer_role_graph(cur_scene)
    if fixed_priors:
        priors = list(initial_priors or [])
    else:
        det_priors = build_role_based_relation_priors(cur_scene, role_graph)
        priors = _merge_relation_priors(cur_scene, role_graph, det_priors, vlm_priors, cfg.vlm_prior_weight_scale) if vlm_priors and cfg.merge_vlm_with_deterministic else (list(vlm_priors) if vlm_priors else det_priors)
    cur_score, cur_metrics, cur_dir, cur_rel, cur_func = _score_scene_full(cur_scene, role_graph, priors, timing)
    return cur_scene, cur_metrics, cur_dir, cur_rel, cur_func, actions, priors, role_graph


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
    initial_role_graph = infer_role_graph(current_scene)
    frozen_deterministic_priors = build_role_based_relation_priors(current_scene, initial_role_graph)
    steps_after_valid_pbl = 0

    for step in range(cfg.max_steps):
        step_t0 = _now()
        step_dir = out_root / f"step_{step:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        _write_json(step_dir / "scene_before.json", current_scene)

        t_render0 = _now()
        respace.render_scene_frame(current_scene, filename=f"step_{step:02d}", pth_viz_output=step_dir)
        diag_path = step_dir / "diag" / f"step_{step:02d}.jpg"
        top_path = render_annotated_top_view(current_scene, f"step_{step:02d}", step_dir, resolution=(1024, 1024), show_assets=True, font_size=14)
        render_sec = _now() - t_render0
        timing_stats.render_sec += render_sec

        role_graph = infer_role_graph(current_scene)
        deterministic_priors = list(frozen_deterministic_priors) if cfg.freeze_deterministic_priors else build_role_based_relation_priors(current_scene, role_graph)

        if cfg.use_vlm_relation_priors:
            need_refresh = relation_priors_cache is None or cfg.refresh_vlm_relation_priors_every_step
            if need_refresh:
                t_rel0 = _now()
                generated = _safe_generate_relation_priors(generator, diag_path, top_path, current_scene, extra_hints_text, cfg.relation_prior_retries, cfg.relation_prior_temperature, cfg.relation_prior_max_tokens, cfg.relation_prior_confidence, step_dir)
                rel_gen_sec = _now() - t_rel0
                timing_stats.vlm_sec += rel_gen_sec
                if generated:
                    relation_priors_cache = generated
                    relation_priors_source = "vlm_relation_priors"
                    print(f"[step {step:02d}] built {len(relation_priors_cache)} VLM relation priors ({rel_gen_sec:.2f}s)")
                else:
                    relation_priors_cache = None
                    relation_priors_source = "deterministic_fallback"
                    print(f"[step {step:02d}] VLM relation priors unavailable, fallback to deterministic")
            current_priors = _merge_relation_priors(current_scene, role_graph, deterministic_priors, relation_priors_cache, cfg.vlm_prior_weight_scale) if cfg.merge_vlm_with_deterministic else list(relation_priors_cache or deterministic_priors)
        else:
            current_priors = deterministic_priors
            relation_priors_cache = None
            relation_priors_source = "deterministic"

        score_before, metrics_before, dir_before, rel_before, func_before = _score_scene_full(current_scene, role_graph, current_priors, timing_stats)
        pbl_before = _get_float_metric(metrics_before, "total_pbl_loss")

        _write_json(step_dir / "role_graph_before.json", {"categories": role_graph.categories, "role_by_idx": role_graph.role_by_idx, "function_by_idx": role_graph.function_by_idx, "zone_by_idx": role_graph.zone_by_idx, "accessory_to_anchor": role_graph.accessory_to_anchor, "notes": role_graph.notes})
        _write_json(step_dir / "relation_priors.json", current_priors)
        _write_json(step_dir / "relation_priors_meta.json", {"source": relation_priors_source, "count": len(current_priors)})
        print(f"[step {step:02d}] relation priors source={relation_priors_source} count={len(current_priors)}")
        print(
            f"[step {step:02d}] pbl={pbl_before:.6f} oob={_get_float_metric(metrics_before, 'total_oob_loss'):.6f} "
            f"mbl={_get_float_metric(metrics_before, 'total_mbl_loss'):.6f} dir={dir_before:.4f} rel={rel_before:.4f} func={func_before:.4f} "
            f"score={score_before:.6f} valid_pbl={pbl_before <= 1e-8}"
        )

        if step > 0 and cfg.stop_when_valid_pbl and pbl_before <= 1e-8:
            if steps_after_valid_pbl >= cfg.max_steps_after_valid_pbl:
                print(f"[step {step:02d}] early stop: valid PBL already preserved for enough cleanup steps")
                break
            if rel_before < 0.35 and dir_before < 0.15 and func_before < 0.45:
                print(f"[step {step:02d}] early stop: scene already good enough")
                break

        if pbl_before <= 1e-8 and cfg.cleanup_only_after_valid_pbl:
            prompt_sec = 0.0
            parse_sec = 0.0
            applied_count = 0
            changes = []
            parse_result = parse_move_prompt("")
            scene_after_prompt = _deepcopy_scene(current_scene)
            _write_text(step_dir / "move_prompt.txt", "[cleanup-only-after-valid-pbl]")
            _write_json(step_dir / "move_prompt_parse.json", {"room_name": "", "header_line": "", "parse_warnings": ["cleanup_only_after_valid_pbl"], "num_edits": 0, "edits": []})
            _write_json(step_dir / "applied_edits.json", {"applied_count": 0, "changes": []})
            work_cfg = copy.deepcopy(cfg)
            work_cfg.max_rounds = min(cfg.max_rounds_after_valid_pbl, cfg.max_rounds)
            work_cfg.max_objects_per_round = min(cfg.max_objects_after_valid_pbl, cfg.max_objects_per_round)
            work_cfg.step_xy = min(cfg.step_xy, 0.16)
            work_cfg.step_yaw = min(cfg.step_yaw, 10.0)
            t_opt0 = _now()
            optimized_scene, metrics_after, dir_after, rel_after, func_after, actions, priors_after, role_graph_after = _optimize_after_prompt(scene_after_prompt, [], work_cfg, initial_priors=current_priors, fixed_priors=False, vlm_priors=relation_priors_cache if cfg.use_vlm_relation_priors and relation_priors_source.startswith("vlm") else None, timing=timing_stats)
        else:
            t_prompt0 = _now()
            prompt_result = _safe_generate_move_prompt(generator, diag_path, top_path, current_scene, extra_hints_text, cfg.move_prompt_retries, cfg.move_prompt_temperature, cfg.move_prompt_max_tokens, step_dir)
            prompt_sec = _now() - t_prompt0
            timing_stats.vlm_sec += prompt_sec
            _write_text(step_dir / "move_prompt.txt", prompt_result.move_prompt)
            print(f"[step {step:02d}] move_prompt generated ({len(prompt_result.move_prompt)} chars), temp={cfg.move_prompt_temperature}")

            t_parse0 = _now()
            parse_result = parse_move_prompt(prompt_result.move_prompt)
            parse_sec = _now() - t_parse0
            _write_json(step_dir / "move_prompt_parse.json", {"room_name": parse_result.room_name, "header_line": parse_result.header_line, "parse_warnings": parse_result.parse_warnings, "num_edits": len(parse_result.edits), "edits": [e.__dict__ for e in parse_result.edits]})

            scene_after_prompt = _deepcopy_scene(current_scene)
            scene_after_prompt, applied_count, changes = apply_edits_to_scene(scene_after_prompt, parse_result.edits)
            _write_json(step_dir / "applied_edits.json", {"applied_count": applied_count, "changes": changes})

            t_opt0 = _now()
            optimized_scene, metrics_after, dir_after, rel_after, func_after, actions, priors_after, role_graph_after = _optimize_after_prompt(scene_after_prompt, parse_result.edits, cfg, initial_priors=current_priors, fixed_priors=False, vlm_priors=relation_priors_cache if cfg.use_vlm_relation_priors and relation_priors_source.startswith("vlm") else None, timing=timing_stats)
        opt_sec = _now() - t_opt0
        timing_stats.optimize_sec += opt_sec
        score_after = _W_PBL * _get_float_metric(metrics_after, "total_pbl_loss") + _W_DIR * dir_after + _W_REL * rel_after + _W_FUNC * func_after

        t_post0 = _now()
        _write_json(step_dir / "optimizer_actions.json", actions)
        _write_json(step_dir / "scene_after.json", optimized_scene)
        _write_json(step_dir / "role_graph_after.json", {"categories": role_graph_after.categories, "role_by_idx": role_graph_after.role_by_idx, "function_by_idx": role_graph_after.function_by_idx, "zone_by_idx": role_graph_after.zone_by_idx, "accessory_to_anchor": role_graph_after.accessory_to_anchor, "notes": role_graph_after.notes})
        _write_json(step_dir / "relation_priors_after.json", priors_after)
        _write_json(step_dir / "relation_priors_after_meta.json", {"source": relation_priors_source, "count": len(priors_after)})
        post_sec = _now() - t_post0

        total_sec = _now() - step_t0
        print(
            f"[step {step:02d}] time total={total_sec:.2f}s (prompt={prompt_sec:.2f}s, parse={parse_sec:.2f}s, opt={opt_sec:.2f}s, post={post_sec:.2f}s) applied={applied_count} "
            f"pbl {pbl_before:.6f}->{_get_float_metric(metrics_after, 'total_pbl_loss'):.6f} "
            f"dir {dir_before:.4f}->{dir_after:.4f} rel {rel_before:.4f}->{rel_after:.4f} func {func_before:.4f}->{func_after:.4f} "
            f"score {score_before:.6f}->{score_after:.6f}"
        )

        pbl_after = _get_float_metric(metrics_after, "total_pbl_loss")
        accepted = False
        reject_reason = "no_improvement"
        if pbl_before <= 1e-8:
            if pbl_after > 1e-8:
                reject_reason = "reintroduced_pbl"
            elif dir_after > dir_before + cfg.max_dir_increase_after_valid_pbl:
                reject_reason = "direction_worse"
            elif rel_after > rel_before + cfg.max_rel_increase_after_valid_pbl:
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
            dir_guard = dir_after <= dir_before + cfg.max_dir_increase_prevalid
            rel_guard = rel_after <= rel_before + cfg.max_rel_increase_prevalid
            func_guard = func_after <= func_before + cfg.max_func_increase_prevalid
            if pbl_improved and score_guard and dir_guard and rel_guard and func_guard:
                accepted = True
            elif score_after < score_before - cfg.monotonic_eps and pbl_after <= pbl_before + 0.005 and func_guard:
                accepted = True
            else:
                if not pbl_improved:
                    reject_reason = "pbl_not_better"
                elif not score_guard:
                    reject_reason = "score_worse_too_much"
                elif not dir_guard:
                    reject_reason = "direction_worse_too_much"
                elif not rel_guard:
                    reject_reason = "relation_worse_too_much"
                else:
                    reject_reason = "functional_worse_too_much"

        if accepted:
            current_scene = optimized_scene
            if cfg.use_vlm_relation_priors and cfg.refresh_vlm_relation_priors_every_step:
                relation_priors_cache = None
            if pbl_after <= 1e-8:
                steps_after_valid_pbl = steps_after_valid_pbl + 1 if pbl_before <= 1e-8 else 0
            else:
                steps_after_valid_pbl = 0
            print(f"[step {step:02d}] accepted")
        else:
            print(f"[step {step:02d}] rejected: {reject_reason}")

        step_runtime_records.append({
            "step": step,
            "runtime_sec": round(total_sec, 4),
            "accepted": accepted,
            "relation_priors_source": relation_priors_source,
            "num_relation_priors": len(priors_after),
            "render_sec": round(render_sec, 4),
            "prompt_sec": round(prompt_sec, 4),
            "opt_sec": round(opt_sec, 4),
            "eval_sec_accum": round(timing_stats.eval_sec, 4),
        })

        if cfg.stop_when_valid_pbl and accepted and _get_float_metric(metrics_after, "total_pbl_loss") <= 1e-8 and dir_after < 0.15 and rel_after < 0.35 and func_after < 0.4:
            break

    if cfg.render_final:
        final_dir = out_root / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        t_render0 = _now()
        respace.render_scene_frame(current_scene, filename="final", pth_viz_output=final_dir)
        render_annotated_top_view(current_scene, "final", final_dir, resolution=(1024, 1024), show_assets=True, font_size=14)
        timing_stats.render_sec += _now() - t_render0
        _write_json(final_dir / "scene.json", current_scene)

    role_graph = infer_role_graph(current_scene)
    deterministic_priors = list(frozen_deterministic_priors) if cfg.freeze_deterministic_priors else build_role_based_relation_priors(current_scene, role_graph)
    if cfg.use_vlm_relation_priors and relation_priors_cache:
        priors = _merge_relation_priors(current_scene, role_graph, deterministic_priors, relation_priors_cache, cfg.vlm_prior_weight_scale) if cfg.merge_vlm_with_deterministic else relation_priors_cache
    else:
        priors = deterministic_priors
        if not cfg.use_vlm_relation_priors:
            relation_priors_source = "deterministic"
    final_score, final_metrics, final_dir_loss, final_rel_loss, final_func_loss = _score_scene_full(current_scene, role_graph, priors, timing_stats)
    summary = {
        "relation_priors_source": relation_priors_source,
        "num_relation_priors": len(priors),
        "total_runtime_sec": round(_now() - overall_t0, 4),
        "render_sec": round(timing_stats.render_sec, 4),
        "vlm_sec": round(timing_stats.vlm_sec, 4),
        "optimize_sec": round(timing_stats.optimize_sec, 4),
        "eval_sec": round(timing_stats.eval_sec, 4),
        "step_runtime_records": step_runtime_records,
        "final_metrics": final_metrics,
        "final_dir_loss": round(final_dir_loss, 4),
        "final_rel_loss": round(final_rel_loss, 4),
        "final_func_loss": round(final_func_loss, 4),
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
        max_dir_increase_after_valid_pbl=float(os.getenv("MAX_DIR_INCREASE_AFTER_VALID_PBL", "0.04")),
        max_rel_increase_after_valid_pbl=float(os.getenv("MAX_REL_INCREASE_AFTER_VALID_PBL", "0.12")),
        max_func_increase_after_valid_pbl=float(os.getenv("MAX_FUNC_INCREASE_AFTER_VALID_PBL", "0.12")),
        max_score_increase_prevalid=float(os.getenv("MAX_SCORE_INCREASE_PREVALID", "0.06")),
        max_dir_increase_prevalid=float(os.getenv("MAX_DIR_INCREASE_PREVALID", "0.12")),
        max_rel_increase_prevalid=float(os.getenv("MAX_REL_INCREASE_PREVALID", "0.55")),
        max_func_increase_prevalid=float(os.getenv("MAX_FUNC_INCREASE_PREVALID", "0.35")),
        render_final=os.getenv("RENDER_FINAL", "1") not in ("0", "false", "False", ""),
        monotonic_eps=float(os.getenv("MONOTONIC_EPS", "1e-12")),
        step_xy=float(os.getenv("STEP_XY", "0.22")),
        step_yaw=float(os.getenv("STEP_YAW", "15.0")),
        anchor_lock_pbl_threshold=float(os.getenv("ANCHOR_LOCK_PBL_THRESHOLD", "0.08")),
        role_refine_blend=float(os.getenv("ROLE_REFINE_BLEND", "0.66")),
    )

    extra_hints_text = (
        "GLOBAL SAFETY CONSTRAINTS:\n"
        "1) All objects must stay fully inside the room.\n"
        "2) Avoid overlaps; keep small but visible clearance.\n"
        "3) Prioritize OOB and collision fixes before aesthetics.\n"
        "4) Preserve dominant-anchor and accessory structure.\n"
        "5) Keep interactive fronts usable and avoid over-crowding one functional anchor."
    )

    model = os.getenv("MOVE_PROMPT_MODEL", os.getenv("YUNWU_AI_MODEL", "gpt-4o"))
    generator = GPTVLMovePromptGeneratorV5(model=model, api_base=os.getenv("YUNWU_AI_API_BASE"), api_key=os.getenv("YUNWU_AI_API_KEY"), timeout_s=float(os.getenv("MOVE_PROMPT_TIMEOUT_S", "120")))

    summary = optimize_scene_refactored(scene=updated_scene, out_root=out_root, respace=ReSpace(), generator=generator, extra_hints_text=extra_hints_text, cfg=cfg)
    print("\n=== Done ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
