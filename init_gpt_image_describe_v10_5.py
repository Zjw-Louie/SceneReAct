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
from scorer.gpt_vl_image_describe_v10 import (
    GPTVLMovePromptGeneratorV5,
    ObjectEdit,
    apply_edits_to_scene,
    parse_move_prompt,
    quaternion_from_yaw,
    yaw_from_quaternion,
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


# ------------------------------
# categories
# ------------------------------

_MODEL_INFO_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


NORMALIZE_CATEGORY_MAP = {
    "king-size bed": "bed",
    "single bed": "bed",
    "double bed": "bed",
    "bunk bed": "bed",
    "kids bed": "bed",
    "children bed": "bed",
    "nightstand": "nightstand",
    "night stand": "nightstand",
    "bedside table": "nightstand",
    "bedside cabinet": "nightstand",
    "wardrobe": "wardrobe",
    "armoire": "wardrobe",
    "desk": "desk",
    "dressing table": "desk",
    "writing desk": "desk",
    "computer desk": "desk",
    "study desk": "desk",
    "chair": "chair",
    "office chair": "chair",
    "desk chair": "chair",
    "dining chair": "chair",
    "armchair": "chair",
    "lounge chair": "chair",
    "stool": "chair",
    "bar stool": "chair",
    "barstool": "chair",
    "sofa": "sofa",
    "couch": "sofa",
    "loveseat": "sofa",
    "coffee table": "coffee table",
    "tea table": "coffee table",
    "tv stand": "tv stand",
    "tv cabinet": "tv stand",
    "media console": "tv stand",
    "cabinet": "cabinet",
    "sideboard": "cabinet",
    "dresser": "cabinet",
    "shelf": "shelf",
    "bookcase": "shelf",
    "bookshelf": "shelf",
    "lamp": "lamp",
    "floor lamp": "lamp",
    "table lamp": "lamp",
    "pendant lamp": "lamp",
    "rug": "rug",
    "carpet": "rug",
    "mirror": "mirror",
    "plant": "plant",
    "potted plant": "plant",
    "curtain": "curtain",
    "table": "table",
}

KEYWORD_CATEGORIES = (
    "bed",
    "nightstand",
    "wardrobe",
    "desk",
    "table",
    "chair",
    "sofa",
    "cabinet",
    "shelf",
    "lamp",
    "rug",
    "mirror",
    "plant",
    "curtain",
)


@dataclass
class Config:
    max_steps: int = 3
    max_rounds: int = 2
    max_objects_per_round: int = 6
    proxy_topk: int = 2
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
    freeze_deterministic_priors: bool = True
    local_repair_passes: int = 1
    full_repair_after_refine_passes: int = 1
    stop_when_valid_pbl: bool = True
    cleanup_only_after_valid_pbl: bool = True
    max_steps_after_valid_pbl: int = 1
    max_objects_after_valid_pbl: int = 3
    max_rounds_after_valid_pbl: int = 1
    min_score_improve_after_valid_pbl: float = 0.03
    max_dir_increase_after_valid_pbl: float = 0.03
    max_rel_increase_after_valid_pbl: float = 0.10
    max_score_increase_prevalid: float = 0.05
    max_dir_increase_prevalid: float = 0.10
    max_rel_increase_prevalid: float = 0.50
    render_final: bool = True
    monotonic_eps: float = 1e-12
    step_xy: float = 0.22
    step_yaw: float = 15.0
    anchor_lock_pbl_threshold: float = 0.08
    role_refine_blend: float = 0.72


_REL_TYPES = {
    "near",
    "distance_band",
    "facing",
    "facing_pair",
    "centered_with",
    "in_front_of",
    "side_of",
    "against_wall",
    "parallel",
}


def _load_model_info() -> Dict[str, Dict[str, Any]]:
    global _MODEL_INFO_CACHE
    if _MODEL_INFO_CACHE is not None:
        return _MODEL_INFO_CACHE

    pth = os.getenv("PTH_3DFUTURE_ASSETS", "")
    candidates = [
        Path(pth) / "model_info.json",
        Path(pth).parent / "model_info.json",
        Path(pth) / ".." / "model_info.json",
    ]
    info_path: Optional[Path] = None
    for candidate in candidates:
        if candidate.exists():
            info_path = candidate
            break

    if info_path is None:
        _MODEL_INFO_CACHE = {}
        return _MODEL_INFO_CACHE

    raw = json.loads(info_path.read_text(encoding="utf-8"))
    result: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            for key in ("model_id", "jid", "id", "modelId"):
                jid = item.get(key)
                if isinstance(jid, str) and jid:
                    result[jid] = item
                    break
    elif isinstance(raw, dict):
        result = raw

    _MODEL_INFO_CACHE = result
    return result


def _normalize_category(category: str) -> str:
    category = category.strip().lower()
    if category in NORMALIZE_CATEGORY_MAP:
        return NORMALIZE_CATEGORY_MAP[category]
    for key, value in NORMALIZE_CATEGORY_MAP.items():
        if key in category:
            return value
    for keyword in KEYWORD_CATEGORIES:
        if keyword in category:
            return keyword
    return category


def _category_from_desc(desc: str) -> str:
    desc = desc.lower()
    for key, value in NORMALIZE_CATEGORY_MAP.items():
        if key in desc:
            return value
    return "unknown"


def _get_obj_category(obj: Dict[str, Any]) -> str:
    jid = obj.get("sampled_asset_jid") or obj.get("jid") or ""
    info = _load_model_info()

    for key in [jid, jid.split("-(")[0] if "-(" in jid else jid]:
        if key in info:
            category = (info[key].get("category") or info[key].get("super-category") or "").strip().lower()
            if category:
                return _normalize_category(category)

    base = jid.split("-(")[0] if "-(" in jid else jid
    if "_" in base:
        prefix = base.rsplit("_", 1)[0]
        if prefix in info:
            category = (info[prefix].get("category") or info[prefix].get("super-category") or "").strip().lower()
            if category:
                return _normalize_category(category)

    desc = obj.get("desc") or obj.get("description") or obj.get("style_description") or ""
    return _category_from_desc(desc)


def _obj_size_xz(obj: Dict[str, Any]) -> Tuple[float, float]:
    size = obj.get("size", [1.0, 1.0, 1.0])
    sx = float(size[0]) if len(size) > 0 else 1.0
    sz = float(size[2]) if len(size) > 2 else 1.0
    return sx, sz


def _distance_to_nearest_wall_xz(scene: Dict[str, Any], pos: Sequence[float]) -> float:
    pts = [(float(p[0]), float(p[2])) for p in scene.get("bounds_bottom", []) if isinstance(p, list) and len(p) >= 3]
    if len(pts) < 3:
        return 999.0
    best = float("inf")
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
        best = min(best, math.hypot(pos[0] - proj_x, pos[2] - proj_z))
    return best


def _refine_categories_contextual(scene: Dict[str, Any], categories: List[str]) -> List[str]:
    objs = scene.get("objects", [])
    refined = list(categories)
    bed_indices = [i for i, cat in enumerate(categories) if cat == "bed"]

    for i, cat in enumerate(categories):
        obj = objs[i]
        sx, sz = _obj_size_xz(obj)
        small_table_like = cat in {"cabinet", "shelf", "desk", "table", "unknown"} and max(sx, sz) <= 0.8
        if small_table_like and bed_indices:
            nearest_bed = min(_xz_dist(obj, objs[b]) for b in bed_indices)
            if nearest_bed <= 1.2:
                refined[i] = "nightstand"
                continue

        if cat in {"table", "desk"}:
            wall_d = _distance_to_nearest_wall_xz(scene, obj.get("pos", [0.0, 0.0, 0.0]))
            long_side = max(sx, sz)
            short_side = min(sx, sz)
            if wall_d < 0.25 and long_side <= 1.4 and short_side <= 0.5:
                refined[i] = "console table"

    return refined


def _build_category_map(scene: Dict[str, Any]) -> List[str]:
    initial = [_get_obj_category(obj) for obj in scene.get("objects", [])]
    return _refine_categories_contextual(scene, initial)


# ------------------------------
# geometry helpers
# ------------------------------

def _normalize_angle(deg: float) -> float:
    return deg % 360.0


def _angle_diff(a: float, b: float) -> float:
    d = abs(_normalize_angle(a) - _normalize_angle(b))
    return min(d, 360.0 - d)


def _xz_dist(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    pa = a.get("pos", [0.0, 0.0, 0.0])
    pb = b.get("pos", [0.0, 0.0, 0.0])
    return math.hypot(pa[0] - pb[0], pa[2] - pb[2])


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


def _room_center_xz(scene: Dict[str, Any]) -> Tuple[float, float]:
    pts = [(float(p[0]), float(p[2])) for p in scene.get("bounds_bottom", []) if isinstance(p, list) and len(p) >= 3]
    if not pts:
        return 0.0, 0.0
    return sum(x for x, _ in pts) / len(pts), sum(z for _, z in pts) / len(pts)


def _find_nearest_wall_yaw(scene: Dict[str, Any], pos: Sequence[float]) -> Optional[float]:
    pts = [(float(p[0]), float(p[2])) for p in scene.get("bounds_bottom", []) if isinstance(p, list) and len(p) >= 3]
    if len(pts) < 3:
        return None
    center_x, center_z = _room_center_xz(scene)
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


def _obj_diag_size_xz(obj: Dict[str, Any]) -> float:
    sx, sz = _obj_size_xz(obj)
    return math.hypot(sx, sz)


def _pair_target_dist(a: Dict[str, Any], b: Dict[str, Any], alpha: float = 0.35, bias: float = 0.15) -> float:
    return alpha * (_obj_diag_size_xz(a) + _obj_diag_size_xz(b)) + bias

def _table_area_xz(obj: Dict[str, Any]) -> float:
    sx, sz = _obj_size_xz(obj)
    return sx * sz


def _main_table_index(scene: Dict[str, Any], categories: List[str]) -> Optional[int]:
    table_indices = [i for i, c in enumerate(categories) if c == "table"]
    if not table_indices:
        return None
    return max(table_indices, key=lambda i: _table_area_xz(scene["objects"][i]))


def _console_anchor_indices(scene: Dict[str, Any], categories: List[str]) -> List[int]:
    result: List[int] = [i for i, c in enumerate(categories) if c == "console table"]
    main_table = _main_table_index(scene, categories)
    for i, c in enumerate(categories):
        if c != "table" or i == main_table:
            continue
        obj = scene["objects"][i]
        sx, sz = _obj_size_xz(obj)
        if _distance_to_nearest_wall_xz(scene, obj.get("pos", [0.0, 0.0, 0.0])) < 0.45 and min(sx, sz) <= 0.75 and sx * sz <= 1.20:
            result.append(i)
    out: List[int] = []
    seen: Set[int] = set()
    for i in result:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _major_anchor_indices(scene: Dict[str, Any], categories: List[str]) -> Set[int]:
    anchors: Set[int] = set(i for i, c in enumerate(categories) if c in {"bed", "desk", "console table"})
    main_table = _main_table_index(scene, categories)
    if main_table is not None:
        anchors.add(main_table)
    return anchors


def _nudge_object_inward_from_wall(scene: Dict[str, Any], idx: int, desired_dist: float) -> None:
    obj = scene.get("objects", [])[idx]
    pos = list(obj.get("pos", [0.0, 0.0, 0.0]))
    cur = _distance_to_nearest_wall_xz(scene, pos)
    if cur >= desired_dist:
        return
    wall_yaw = _find_nearest_wall_yaw(scene, pos)
    if wall_yaw is None:
        return
    nx, nz = _forward_vec_from_yaw(wall_yaw)
    push = desired_dist - cur
    obj["pos"] = [pos[0] + nx * push, pos[1], pos[2] + nz * push]


def _main_table_side_slot(scene: Dict[str, Any], anchor: Dict[str, Any], chair: Dict[str, Any], sign: float) -> Tuple[List[float], float]:
    ax, ay, az = anchor.get("pos", [0.0, 0.0, 0.0])
    ayaw = yaw_from_quaternion(anchor.get("rot", [0.0, 0.0, 0.0, 1.0]))
    fx, fz = _forward_vec_from_yaw(ayaw)
    dist = max(0.55, _pair_target_dist(chair, anchor, alpha=0.46, bias=0.24))
    target_pos = [ax + sign * fx * dist, chair.get("pos", [0.0, 0.0, 0.0])[1], az + sign * fz * dist]
    target_yaw = _normalize_angle(math.degrees(math.atan2(ax - target_pos[0], az - target_pos[2])))
    return target_pos, target_yaw


def _role_refine_layout(scene: Dict[str, Any], categories: List[str], *, blend: float = 0.72) -> None:
    objs = scene.get("objects", [])
    main_table = _main_table_index(scene, categories)
    console_indices = _console_anchor_indices(scene, categories)

    if main_table is not None:
        yaw = yaw_from_quaternion(objs[main_table].get("rot", [0.0, 0.0, 0.0, 1.0]))
        target = _nearest_parallel_wall_yaw(scene, objs[main_table].get("pos", [0.0, 0.0, 0.0]), yaw)
        if target is not None:
            objs[main_table]["rot"] = quaternion_from_yaw(target)
        _nudge_object_inward_from_wall(scene, main_table, desired_dist=0.55)
        _repair_object_local(scene, main_table, passes=1)

    for idx in console_indices:
        yaw = yaw_from_quaternion(objs[idx].get("rot", [0.0, 0.0, 0.0, 1.0]))
        target = _nearest_parallel_wall_yaw(scene, objs[idx].get("pos", [0.0, 0.0, 0.0]), yaw)
        if target is not None:
            objs[idx]["rot"] = quaternion_from_yaw(target)
        _repair_object_local(scene, idx, passes=1)

    chair_assignment = _assign_chairs_to_anchors(scene, categories)

    # console chairs first: one chair in front of each wall console
    for chair_idx, anchor_idx in list(chair_assignment.items()):
        if anchor_idx not in console_indices:
            continue
        target_pos, target_yaw = _slot_pose_for_anchor(scene, objs[anchor_idx], objs[chair_idx], objs, chair_idx, anchor_cat="console table")
        chair = objs[chair_idx]
        cur = chair.get("pos", [0.0, 0.0, 0.0])
        chair["pos"] = [
            (1.0 - blend) * cur[0] + blend * target_pos[0],
            cur[1],
            (1.0 - blend) * cur[2] + blend * target_pos[2],
        ]
        chair["rot"] = quaternion_from_yaw(target_yaw)
        _repair_object_local(scene, chair_idx, passes=1)

    # main-table chairs: prefer two opposite side slots along the table front/back axis
    if main_table is not None:
        assigned = [ci for ci, ai in chair_assignment.items() if ai == main_table]
        if assigned:
            ayaw = yaw_from_quaternion(objs[main_table].get("rot", [0.0, 0.0, 0.0, 1.0]))
            fx, fz = _forward_vec_from_yaw(ayaw)
            ranked = sorted(assigned, key=lambda ci: _signed_proj(objs[main_table]["pos"][0], objs[main_table]["pos"][2], objs[ci]["pos"][0], objs[ci]["pos"][2], fx, fz))
            signs = [-1.0] if len(ranked) == 1 else [-1.0, 1.0]
            for ci, sign in zip(ranked, signs):
                target_pos, target_yaw = _main_table_side_slot(scene, objs[main_table], objs[ci], sign)
                chair = objs[ci]
                cur = chair.get("pos", [0.0, 0.0, 0.0])
                chair["pos"] = [
                    (1.0 - blend) * cur[0] + blend * target_pos[0],
                    cur[1],
                    (1.0 - blend) * cur[2] + blend * target_pos[2],
                ]
                chair["rot"] = quaternion_from_yaw(target_yaw)
                _repair_object_local(scene, ci, passes=1)


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
# relation priors and functional slots
# ------------------------------

def _anchor_capacity(category: str) -> int:
    if category == "desk":
        return 1
    if category == "table":
        return 2
    if category == "console table":
        return 1
    if category == "sofa":
        return 1
    return 1


def _choose_open_side_for_anchor(
    scene: Dict[str, Any],
    anchor: Dict[str, Any],
    objects: List[Dict[str, Any]],
    exclude_idx: int,
    anchor_cat: Optional[str] = None,
) -> Tuple[float, float]:
    pos = anchor.get("pos", [0.0, 0.0, 0.0])
    yaw = yaw_from_quaternion(anchor.get("rot", [0.0, 0.0, 0.0, 1.0]))
    fx, fz = _forward_vec_from_yaw(yaw)
    lx, lz = fz, -fx

    if anchor_cat == "desk":
        candidate_specs = [((fx, fz), 0.9), ((lx, lz), 0.15), ((-lx, -lz), 0.15), ((-fx, -fz), -0.5)]
    elif anchor_cat == "sofa":
        candidate_specs = [((fx, fz), 0.6), ((lx, lz), 0.1), ((-lx, -lz), 0.1), ((-fx, -fz), -0.3)]
    else:
        candidate_specs = [((fx, fz), 0.2), ((-fx, -fz), 0.0), ((lx, lz), 0.1), ((-lx, -lz), 0.1)]

    best_vec = candidate_specs[0][0]
    best_score = -1e9
    room_cx, room_cz = _room_center_xz(scene)
    floor_poly = _get_floor_polygon(scene)
    for (vx, vz), semantic_bias in candidate_specs:
        sample_x = pos[0] + vx * 0.7
        sample_z = pos[2] + vz * 0.7
        room_bonus = 0.0
        if floor_poly is not None and floor_poly.contains(ShapelyPoint(sample_x, sample_z)):
            room_bonus += 1.5
        wall_d = _distance_to_nearest_wall_xz(scene, [sample_x, 0.0, sample_z])
        open_bonus = min(1.5, wall_d)
        occ_penalty = 0.0
        for j, other in enumerate(objects):
            if j == exclude_idx:
                continue
            d = math.hypot(sample_x - other.get("pos", [0.0, 0.0, 0.0])[0], sample_z - other.get("pos", [0.0, 0.0, 0.0])[2])
            if d < 0.9:
                occ_penalty += (0.9 - d)
        center_bonus = 0.2 * ((room_cx - pos[0]) * vx + (room_cz - pos[2]) * vz)
        score = semantic_bias + room_bonus + open_bonus + center_bonus - occ_penalty
        if score > best_score:
            best_score = score
            best_vec = (vx, vz)
    return best_vec


def _chair_target_yaw_for_anchor(
    anchor: Dict[str, Any],
    chair_pos: Sequence[float],
    anchor_cat: Optional[str],
    current_yaw: Optional[float] = None,
) -> float:
    apos = anchor.get("pos", [0.0, 0.0, 0.0])
    face_center_yaw = _normalize_angle(math.degrees(math.atan2(apos[0] - chair_pos[0], apos[2] - chair_pos[2])))
    anchor_yaw = yaw_from_quaternion(anchor.get("rot", [0.0, 0.0, 0.0, 1.0]))

    if anchor_cat == "desk":
        candidates = [_normalize_angle(anchor_yaw + 180.0), face_center_yaw]
    elif anchor_cat == "sofa":
        candidates = [face_center_yaw, _normalize_angle(anchor_yaw + 180.0)]
    else:
        candidates = [face_center_yaw]

    if current_yaw is None:
        return candidates[0]
    return min(candidates, key=lambda y: _angle_diff(current_yaw, y))


def _slot_pose_for_anchor(
    scene: Dict[str, Any],
    anchor: Dict[str, Any],
    chair: Dict[str, Any],
    objects: List[Dict[str, Any]],
    exclude_idx: int,
    anchor_cat: Optional[str] = None,
) -> Tuple[List[float], float]:
    ax, ay, az = anchor.get("pos", [0.0, 0.0, 0.0])
    vx, vz = _choose_open_side_for_anchor(scene, anchor, objects, exclude_idx, anchor_cat=anchor_cat)
    dist = max(0.45, _pair_target_dist(chair, anchor, alpha=0.45, bias=0.2))
    target_pos = [ax + vx * dist, chair.get("pos", [0.0, 0.0, 0.0])[1], az + vz * dist]
    current_yaw = yaw_from_quaternion(chair.get("rot", [0.0, 0.0, 0.0, 1.0]))
    target_yaw = _chair_target_yaw_for_anchor(anchor, target_pos, anchor_cat, current_yaw=current_yaw)
    return target_pos, target_yaw


def _assign_chairs_to_anchors(scene: Dict[str, Any], categories: List[str]) -> Dict[int, int]:
    objs = scene.get("objects", [])
    chair_indices = [i for i, c in enumerate(categories) if c == "chair"]
    anchor_indices = [i for i, c in enumerate(categories) if c in {"desk", "table", "sofa"}]

    capacities = {i: _anchor_capacity(categories[i]) for i in anchor_indices}
    assignment: Dict[int, int] = {}
    used: Dict[int, int] = {i: 0 for i in anchor_indices}

    pairs: List[Tuple[float, int, int, int]] = []
    for chair_idx in chair_indices:
        chair = objs[chair_idx]
        for anchor_idx in anchor_indices:
            anchor = objs[anchor_idx]
            d = _xz_dist(chair, anchor)
            penalty = 0.0
            if categories[anchor_idx] == "sofa":
                penalty += 0.4
            pairs.append((d + penalty, chair_idx, anchor_idx, used.get(anchor_idx, 0)))
    pairs.sort(key=lambda x: x[0])

    for _, chair_idx, anchor_idx, _ in pairs:
        if chair_idx in assignment:
            continue
        if used[anchor_idx] >= capacities[anchor_idx]:
            continue
        assignment[chair_idx] = anchor_idx
        used[anchor_idx] += 1

    return assignment


def _build_deterministic_relation_priors(scene: Dict[str, Any], categories: List[str]) -> List[Dict[str, Any]]:
    objs = scene.get("objects", [])
    priors: List[Dict[str, Any]] = []

    # bed -> wall, nightstand -> bed
    bed_indices = [i for i, cat in enumerate(categories) if cat == "bed"]
    nightstand_indices = [i for i, cat in enumerate(categories) if cat == "nightstand"]
    for bed_idx in bed_indices:
        priors.append({"src_idx": bed_idx, "type": "against_wall", "confidence": 0.95, "weight": 1.2})
        nearby = sorted(nightstand_indices, key=lambda j: _xz_dist(objs[bed_idx], objs[j]))[:2]
        for j in nearby:
            if _xz_dist(objs[bed_idx], objs[j]) <= 1.8:
                priors.append({"src_idx": j, "tgt_idx": bed_idx, "type": "near", "confidence": 0.95, "weight": 1.2})
                priors.append({"src_idx": j, "tgt_idx": bed_idx, "type": "side_of", "confidence": 0.9, "weight": 1.0})

    # chairs -> assigned anchors
    chair_assignment = _assign_chairs_to_anchors(scene, categories)
    for chair_idx, anchor_idx in chair_assignment.items():
        priors.append({"src_idx": chair_idx, "tgt_idx": anchor_idx, "type": "near", "confidence": 0.95, "weight": 1.2})
        anchor_cat = categories[anchor_idx]
        if anchor_cat == "desk":
            priors.append({"src_idx": chair_idx, "tgt_idx": anchor_idx, "type": "centered_with", "confidence": 0.9, "weight": 0.9})
        else:
            priors.append({"src_idx": chair_idx, "tgt_idx": anchor_idx, "type": "facing", "confidence": 0.95, "weight": 1.2})

    # storage / wall-aligned anchors -> wall
    for i, cat in enumerate(categories):
        if cat in {"wardrobe", "cabinet", "shelf", "tv stand", "console table"}:
            priors.append({"src_idx": i, "type": "against_wall", "confidence": 0.85, "weight": 0.8})
        elif cat == "table":
            priors.append({"src_idx": i, "type": "parallel", "confidence": 0.8, "weight": 0.9})

    # coffee table -> sofa
    coffee_indices = [i for i, cat in enumerate(categories) if cat == "coffee table"]
    sofa_indices = [i for i, cat in enumerate(categories) if cat == "sofa"]
    for coffee_idx in coffee_indices:
        if not sofa_indices:
            continue
        sofa_idx = min(sofa_indices, key=lambda j: _xz_dist(objs[coffee_idx], objs[j]))
        priors.append({"src_idx": coffee_idx, "tgt_idx": sofa_idx, "type": "near", "confidence": 0.9, "weight": 1.2})
        priors.append({"src_idx": coffee_idx, "tgt_idx": sofa_idx, "type": "in_front_of", "confidence": 0.8, "weight": 1.0})

    unique: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for item in priors:
        key = (item.get("src_idx"), item.get("tgt_idx"), item.get("type"))
        old = unique.get(key)
        if old is None or float(item.get("confidence", 0.0)) > float(old.get("confidence", 0.0)):
            unique[key] = item
    return list(unique.values())


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
    categories: List[str],
    deterministic_priors: List[Dict[str, Any]],
    vlm_priors: Optional[List[Dict[str, Any]]],
    weight_scale: float,
) -> List[Dict[str, Any]]:
    if not vlm_priors:
        return deterministic_priors

    chair_assignment = _assign_chairs_to_anchors(scene, categories)
    merged = list(deterministic_priors)
    for item in vlm_priors:
        if not isinstance(item, dict):
            continue
        src_idx = item.get("src_idx")
        tgt_idx = item.get("tgt_idx")
        rel_type = str(item.get("type", "")).strip()
        if rel_type not in _REL_TYPES:
            continue
        if not isinstance(src_idx, int) or not (0 <= src_idx < len(categories)):
            continue
        src_cat = categories[src_idx]
        tgt_cat = categories[tgt_idx] if isinstance(tgt_idx, int) and 0 <= tgt_idx < len(categories) else "wall"

        # filter obvious VLM drift for this room type
        if src_cat == "chair" and rel_type in {"near", "facing", "centered_with", "in_front_of", "side_of", "distance_band"}:
            assigned = chair_assignment.get(src_idx)
            if assigned is not None and tgt_idx != assigned:
                continue
            if assigned is not None and rel_type == "facing" and categories[assigned] == "desk":
                continue
        if src_cat == "bed" and rel_type in {"facing", "facing_pair", "in_front_of"}:
            continue
        if src_cat in {"table", "desk", "console table"} and rel_type in {"facing", "facing_pair"}:
            continue
        if src_cat == "table" and rel_type == "against_wall":
            continue
        if src_cat in {"nightstand"} and tgt_cat not in {"bed", "wall"}:
            continue

        merged.append({
            "src_idx": src_idx,
            "tgt_idx": tgt_idx,
            "type": rel_type,
            "confidence": min(0.9, float(item.get("confidence", 1.0))),
            "weight": float(item.get("weight", 1.0)) * weight_scale,
            "reason": str(item.get("reason", "")),
        })
    return _dedup_priors(merged)


def _post_refine_functional_layout(scene: Dict[str, Any], categories: List[str], relation_priors: List[Dict[str, Any]], *, full_repair_passes: int = 1) -> None:
    objs = scene.get("objects", [])

    # place chairs into canonical slots for their anchors
    chair_assignment = _assign_chairs_to_anchors(scene, categories)
    for chair_idx, anchor_idx in chair_assignment.items():
        target_pos, target_yaw = _slot_pose_for_anchor(scene, objs[anchor_idx], objs[chair_idx], objs, chair_idx, anchor_cat=categories[anchor_idx])
        chair = objs[chair_idx]
        cur = chair.get("pos", [0.0, 0.0, 0.0])
        chair["pos"] = [0.5 * cur[0] + 0.5 * target_pos[0], cur[1], 0.5 * cur[2] + 0.5 * target_pos[2]]
        chair["rot"] = quaternion_from_yaw(target_yaw)
        _repair_object_local(scene, chair_idx, passes=1)

    # snap major anchors to stable canonical yaw before placing accessories
    for i, cat in enumerate(categories):
        if cat == "table":
            yaw = yaw_from_quaternion(objs[i].get("rot", [0.0, 0.0, 0.0, 1.0]))
            target = _nearest_parallel_wall_yaw(scene, objs[i].get("pos", [0.0, 0.0, 0.0]), yaw)
            if target is not None:
                objs[i]["rot"] = quaternion_from_yaw(target)
                _repair_object_local(scene, i, passes=1)
        elif cat == "bed":
            yaw = yaw_from_quaternion(objs[i].get("rot", [0.0, 0.0, 0.0, 1.0]))
            target = _nearest_normal_axis_yaw(scene, objs[i].get("pos", [0.0, 0.0, 0.0]), yaw)
            if target is not None:
                objs[i]["rot"] = quaternion_from_yaw(target)
                _repair_object_local(scene, i, passes=1)

    # gently symmetrize paired nightstands around bed
    bed_indices = [i for i, c in enumerate(categories) if c == "bed"]
    nightstand_indices = [i for i, c in enumerate(categories) if c == "nightstand"]
    for bed_idx in bed_indices:
        bed = objs[bed_idx]
        bed_yaw = yaw_from_quaternion(bed.get("rot", [0.0, 0.0, 0.0, 1.0]))
        fx, fz = _forward_vec_from_yaw(bed_yaw)
        lx, lz = fz, -fx
        nearby = [j for j in nightstand_indices if _xz_dist(bed, objs[j]) <= 1.8]
        nearby = sorted(nearby, key=lambda j: _xz_dist(bed, objs[j]))[:2]
        if len(nearby) != 2:
            continue
        bx, by, bz = bed.get("pos", [0.0, 0.0, 0.0])
        offset = max(0.6, 0.55 * max(_obj_size_xz(bed)))
        for sign, j in [(-1.0, nearby[0]), (1.0, nearby[1])]:
            target = [bx + sign * lx * offset, objs[j].get("pos", [0.0, 0.0, 0.0])[1], bz + sign * lz * offset]
            cur = objs[j].get("pos", [0.0, 0.0, 0.0])
            objs[j]["pos"] = [0.5 * cur[0] + 0.5 * target[0], cur[1], 0.5 * cur[2] + 0.5 * target[2]]
            _repair_object_local(scene, j, passes=1)

    _ = relation_priors
    if full_repair_passes > 0:
        for i in range(len(objs)):
            _repair_object_local(scene, i, passes=full_repair_passes)


# ------------------------------
# losses
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
    dist_loss = max(0.0, _distance_to_nearest_wall_xz(scene, pos) - 0.3)
    yaw = yaw_from_quaternion(obj.get("rot", [0.0, 0.0, 0.0, 1.0]))
    if category == "bed":
        yaw_loss = _axis_angle_diff(yaw, wall_yaw) / 180.0
    elif category == "table":
        parallel = _nearest_parallel_wall_yaw(scene, pos, yaw)
        yaw_loss = 0.0 if parallel is None else _angle_diff(yaw, parallel) / 180.0
    else:
        yaw_loss = _angle_diff(yaw, _normalize_angle(wall_yaw + 180.0)) / 180.0
    return (dist_loss + 0.5 * yaw_loss) * weight


def _loss_parallel(scene: Dict[str, Any], obj: Dict[str, Any], weight: float, *, category: Optional[str] = None) -> float:
    pos = obj.get("pos", [0.0, 0.0, 0.0])
    yaw = yaw_from_quaternion(obj.get("rot", [0.0, 0.0, 0.0, 1.0]))
    parallel = _nearest_parallel_wall_yaw(scene, pos, yaw)
    if parallel is None:
        return 0.0
    diff = _axis_angle_diff(yaw, parallel) if category == "table" else _angle_diff(yaw, parallel)
    return (diff / 180.0) * weight


def _compute_direction_loss(scene: Dict[str, Any], categories: List[str]) -> Tuple[float, List[Dict[str, Any]]]:
    objs = scene.get("objects", [])
    violations: List[Dict[str, Any]] = []
    room_cx, room_cz = _room_center_xz(scene)
    total = 0.0

    for i, obj in enumerate(objs):
        cat = categories[i]
        yaw = yaw_from_quaternion(obj.get("rot", [0.0, 0.0, 0.0, 1.0]))
        pos = obj.get("pos", [0.0, 0.0, 0.0])
        target_yaw: Optional[float] = None
        threshold = 25.0
        weight = 1.0

        if cat == "bed":
            target_yaw = _nearest_normal_axis_yaw(scene, pos, yaw)
            if target_yaw is not None:
                threshold = 10.0
                weight = 1.0
        elif cat == "chair":
            anchor_assignment = _assign_chairs_to_anchors(scene, categories)
            if i in anchor_assignment:
                anchor_idx = anchor_assignment[i]
                target_yaw = _chair_target_yaw_for_anchor(
                    objs[anchor_idx],
                    pos,
                    categories[anchor_idx],
                    current_yaw=yaw,
                )
                threshold = 22.0 if categories[anchor_idx] == "desk" else 20.0
                weight = 0.9 if categories[anchor_idx] == "desk" else 1.0
        elif cat in {"wardrobe", "cabinet", "shelf", "console table", "tv stand"}:
            target_yaw = _normalize_angle(math.degrees(math.atan2(room_cx - pos[0], room_cz - pos[2])))
            threshold = 45.0
            weight = 0.5
        elif cat == "desk":
            wall_yaw = _find_nearest_wall_yaw(scene, pos)
            if wall_yaw is not None:
                target_yaw = _normalize_angle(wall_yaw + 180.0)
                threshold = 25.0
                weight = 0.8
        elif cat == "table":
            parallel_yaw = _nearest_parallel_wall_yaw(scene, pos, yaw)
            if parallel_yaw is not None:
                target_yaw = parallel_yaw
                threshold = 5.0
                weight = 1.75

        if target_yaw is None:
            continue
        diff = _angle_diff(yaw, target_yaw)
        if diff > threshold:
            penalty = diff / 180.0 * weight
            total += penalty
            violations.append({"idx": i, "cat": cat, "current_yaw": round(yaw, 1), "target_yaw": round(target_yaw, 1), "penalty": round(penalty, 4)})

    return total, violations


def _compute_relation_loss(scene: Dict[str, Any], categories: List[str], priors: List[Dict[str, Any]]) -> Tuple[float, List[Dict[str, Any]]]:
    objs = scene.get("objects", [])
    total = 0.0
    violations: List[Dict[str, Any]] = []

    for item in priors:
        rel_type = str(item.get("type", "")).strip()
        if rel_type not in _REL_TYPES:
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
            loss = _loss_against_wall(scene, src, weight, category=categories[src_idx])
        elif rel_type == "parallel":
            loss = _loss_parallel(scene, src, weight, category=categories[src_idx])
        else:
            if not isinstance(tgt_idx, int) or not (0 <= tgt_idx < len(objs)) or tgt_idx == src_idx:
                continue
            tgt = objs[tgt_idx]
            dist = _xz_dist(src, tgt)

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

_W_PBL = 0.6
_W_DIR = 0.2
_W_REL = 0.2


def _get_float_metric(metrics: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else default


def _score_scene_full(
    scene: Dict[str, Any],
    categories: Optional[List[str]] = None,
    relation_priors: Optional[List[Dict[str, Any]]] = None,
    timing_stats: Optional[TimingStats] = None,
) -> Tuple[float, Dict[str, Any], float, float]:
    t0 = time.perf_counter()
    metrics = eval_scene(scene, is_debug=False)
    if categories is None:
        categories = _build_category_map(scene)
    priors = relation_priors or []
    dir_loss, _ = _compute_direction_loss(scene, categories)
    rel_loss, _ = _compute_relation_loss(scene, categories, priors)
    score = _W_PBL * _get_float_metric(metrics, "total_pbl_loss") + _W_DIR * dir_loss + _W_REL * rel_loss
    if timing_stats is not None:
        timing_stats.eval_sec += time.perf_counter() - t0
    return score, metrics, dir_loss, rel_loss



def _collision_area_for_object(scene: Dict[str, Any], idx: int) -> float:
    return sum(area for _, _, _, area in _collision_neighbors(scene, idx))


def _direction_penalty_for_object(scene: Dict[str, Any], categories: List[str], idx: int) -> float:
    total, violations = _compute_direction_loss(scene, categories)
    _ = total
    return sum(v["penalty"] for v in violations if v.get("idx") == idx)


def _relation_penalty_for_object(scene: Dict[str, Any], categories: List[str], priors: List[Dict[str, Any]], idx: int) -> float:
    total, violations = _compute_relation_loss(scene, categories, priors)
    _ = total
    return sum(v["penalty"] for v in violations if v.get("src_idx") == idx or v.get("tgt_idx") == idx)


def _quick_candidate_proxy_score(scene: Dict[str, Any], categories: List[str], priors: List[Dict[str, Any]], idx: int) -> float:
    floor_polygon = _get_floor_polygon(scene)
    oob = 0.0
    if floor_polygon is not None:
        oob = compute_oob(scene["objects"][idx], floor_polygon, scene.get("bounds_bottom", []), scene.get("bounds_top", []), is_debug=False)
    collision = _collision_area_for_object(scene, idx)
    direction = _direction_penalty_for_object(scene, categories, idx)
    relation = _relation_penalty_for_object(scene, categories, priors, idx)
    return 3.0 * oob + 2.0 * collision + 1.0 * direction + 1.2 * relation


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


def _anchor_pose_candidate(scene: Dict[str, Any], categories: List[str], idx: int) -> Optional[Dict[str, Any]]:
    objs = scene.get("objects", [])
    cat = categories[idx]
    obj = objs[idx]
    pos = obj.get("pos", [0.0, 0.0, 0.0])

    if cat == "bed":
        yaw_abs = _nearest_normal_axis_yaw(scene, pos, yaw_from_quaternion(obj.get("rot", [0.0, 0.0, 0.0, 1.0])))
        if yaw_abs is None:
            return None
        return {"kind": "bed_align_wall", "dx": 0.0, "dz": 0.0, "yaw_abs": yaw_abs}

    if cat == "chair":
        chair_assignment = _assign_chairs_to_anchors(scene, categories)
        anchor_idx = chair_assignment.get(idx)
        if anchor_idx is None:
            return None
        target_pos, target_yaw = _slot_pose_for_anchor(scene, objs[anchor_idx], obj, objs, idx, anchor_cat=categories[anchor_idx])
        return {"kind": "chair_slot", "dx": target_pos[0] - pos[0], "dz": target_pos[2] - pos[2], "yaw_abs": target_yaw}

    if cat in {"nightstand"}:
        bed_indices = [j for j, c in enumerate(categories) if c == "bed"]
        if not bed_indices:
            return None
        bed_idx = min(bed_indices, key=lambda j: _xz_dist(obj, objs[j]))
        bed = objs[bed_idx]
        bed_yaw = yaw_from_quaternion(bed.get("rot", [0.0, 0.0, 0.0, 1.0]))
        fx, fz = _forward_vec_from_yaw(bed_yaw)
        lx, lz = fz, -fx
        sign = -1.0 if _signed_proj(bed["pos"][0], bed["pos"][2], pos[0], pos[2], lx, lz) < 0 else 1.0
        offset = max(0.6, 0.55 * max(_obj_size_xz(bed)))
        target_x = bed["pos"][0] + sign * lx * offset
        target_z = bed["pos"][2] + sign * lz * offset
        return {"kind": "nightstand_side", "dx": target_x - pos[0], "dz": target_z - pos[2], "yaw_abs": None}

    if cat in {"wardrobe", "cabinet", "shelf", "tv stand", "console table", "desk"}:
        wall_yaw = _find_nearest_wall_yaw(scene, pos)
        if wall_yaw is None:
            return None
        yaw_abs = _normalize_angle(wall_yaw + 180.0)
        return {"kind": "storage_wall", "dx": 0.0, "dz": 0.0, "yaw_abs": yaw_abs}

    if cat == "table":
        yaw_abs = _nearest_parallel_wall_yaw(scene, pos, yaw_from_quaternion(obj.get("rot", [0.0, 0.0, 0.0, 1.0])))
        if yaw_abs is None:
            return None
        return {"kind": "table_parallel_wall", "dx": 0.0, "dz": 0.0, "yaw_abs": yaw_abs}

    return None


def _generate_candidates(scene: Dict[str, Any], categories: List[str], idx: int, bias_edit: Optional[ObjectEdit], cfg: Config) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    obj = scene["objects"][idx]
    cat = categories[idx]
    step_xy = cfg.step_xy * max(0.8, min(1.6, _obj_diag_size_xz(obj)))
    step_yaw = cfg.step_yaw

    if bias_edit is not None:
        yaw_abs = None if bias_edit.target_yaw_deg is None else float(bias_edit.target_yaw_deg)
        dyaw = 0.0 if bias_edit.no_rotation else float(bias_edit.relative_yaw_deg or 0.0)
        if cat == "table":
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
        candidates.append(
            {
                "kind": "gpt_bias",
                "dx": float(bias_edit.dx),
                "dz": float(bias_edit.dz),
                "dyaw": dyaw,
                "yaw_abs": yaw_abs,
            }
        )

    anchor_cand = _anchor_pose_candidate(scene, categories, idx)
    if anchor_cand is not None:
        anchor_cand.setdefault("dyaw", 0.0)
        candidates.append(anchor_cand)

    allow_free_yaw = cat not in {"table", "bed"}
    allow_diag = cat not in {"table"}
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


def _prioritized_object_indices(scene: Dict[str, Any], categories: List[str], edits: List[ObjectEdit], metrics: Dict[str, Any], cfg: Config) -> List[int]:
    priority: List[int] = []
    hotspot = metrics.get("obj_with_highest_pbl_loss", {})
    hotspot_idx = hotspot.get("idx") if isinstance(hotspot, dict) else None
    if isinstance(hotspot_idx, int):
        priority.append(int(hotspot_idx))
    for edit in edits:
        idx = _find_obj_idx_for_edit(scene, edit)
        if idx is not None:
            priority.append(idx)

    pbl = _get_float_metric(metrics, "total_pbl_loss")
    locked = _major_anchor_indices(scene, categories) if pbl <= cfg.anchor_lock_pbl_threshold else set()

    for i, cat in enumerate(categories):
        if i in locked and i != hotspot_idx:
            continue
        if cat in {"chair", "nightstand", "bed", "desk", "table", "console table"}:
            priority.append(i)
    unique: List[int] = []
    seen: Set[int] = set()
    for idx in priority:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    return unique[: cfg.max_objects_per_round]


# ------------------------------
# optimization per step
# ------------------------------

def _safe_generate_move_prompt(
    generator: GPTVLMovePromptGeneratorV5,
    diag_path: Path,
    top_path: Path,
    scene: Dict[str, Any],
    extra_context: str,
    retries: int,
    temperature: float,
    max_tokens: int,
    trial_dir: Path,
):
    last_exc: Optional[BaseException] = None
    for retry in range(retries):
        try:
            return generator.generate(
                diag_image_path=diag_path,
                top_image_path=top_path,
                scene=scene,
                extra_context=extra_context,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            last_exc = exc
            _write_text(trial_dir / f"move_prompt_retry{retry+1}.error.txt", traceback.format_exc())
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("move prompt generation failed")


def _clean_relation_priors(
    priors: List[Dict[str, Any]],
    n_obj: int,
    min_confidence: float,
) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    per_src_count: Dict[int, int] = {}

    for item in priors:
        if not isinstance(item, dict):
            continue
        rel_type = str(item.get("type", "")).strip()
        if rel_type not in _REL_TYPES:
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

        cleaned.append(
            {
                "src_idx": src_idx,
                "tgt_idx": tgt_idx,
                "type": rel_type,
                "confidence": confidence,
                "weight": float(item.get("weight", 1.0)),
                "reason": str(item.get("reason", "")),
            }
        )
        per_src_count[src_idx] = per_src_count.get(src_idx, 0) + 1
        if len(cleaned) >= 24:
            break

    return cleaned


def _safe_generate_relation_priors(
    generator: GPTVLMovePromptGeneratorV5,
    diag_path: Path,
    top_path: Path,
    scene: Dict[str, Any],
    extra_context: str,
    retries: int,
    temperature: float,
    max_tokens: int,
    min_confidence: float,
    out_dir: Path,
) -> Optional[List[Dict[str, Any]]]:
    if not hasattr(generator, "generate_relation_priors"):
        _write_text(
            out_dir / "vlm_relation_priors.error.txt",
            "generator does not implement generate_relation_priors(); fallback to deterministic relation priors.\n",
        )
        return None

    last_exc: Optional[BaseException] = None
    for retry in range(retries):
        try:
            result = generator.generate_relation_priors(
                diag_image_path=diag_path,
                top_image_path=top_path,
                scene=scene,
                extra_context=extra_context,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw = json.loads(result.json_text)
            cleaned = _clean_relation_priors(raw.get("relations", []), len(scene.get("objects", [])), min_confidence)
            _write_json(
                out_dir / "vlm_relation_priors.json",
                {
                    "raw_text": result.raw_text,
                    "raw_json": raw,
                    "cleaned_relations": cleaned,
                },
            )
            return cleaned
        except Exception as exc:
            last_exc = exc
            _write_text(out_dir / f"vlm_relation_priors_retry{retry+1}.error.txt", traceback.format_exc())
    if last_exc is not None:
        return None
    return None


def _evaluate_best_local_move(
    scene: Dict[str, Any],
    categories: List[str],
    priors: List[Dict[str, Any]],
    idx: int,
    bias_edit: Optional[ObjectEdit],
    current_score: float,
    cfg: Config,
    *,
    fixed_priors: bool,
    timing: Optional[TimingStats] = None,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], float, float, float, Dict[str, Any]]]:
    candidates = _generate_candidates(scene, categories, idx, bias_edit, cfg)
    proxy_ranked: List[Tuple[float, Dict[str, Any], Dict[str, Any], List[str]]] = []

    for cand in candidates:
        sc = _apply_delta(
            scene,
            idx,
            dx=float(cand.get("dx", 0.0)),
            dz=float(cand.get("dz", 0.0)),
            dyaw=float(cand.get("dyaw", 0.0)),
            yaw_abs=None if cand.get("yaw_abs") is None else float(cand.get("yaw_abs")),
        )
        _repair_object_local(sc, idx, cfg.local_repair_passes)
        local_categories = _build_category_map(sc)
        proxy = _quick_candidate_proxy_score(sc, local_categories, priors, idx)
        proxy_ranked.append((proxy, cand, sc, local_categories))

    proxy_ranked.sort(key=lambda x: x[0])
    best: Optional[Tuple[Dict[str, Any], Dict[str, Any], float, float, float, Dict[str, Any]]] = None

    for proxy, cand, sc, local_categories in proxy_ranked[: max(1, cfg.proxy_topk)]:
        local_priors = priors if fixed_priors else _build_deterministic_relation_priors(sc, local_categories)
        score, metrics, dir_loss, rel_loss = _score_scene_full(sc, local_categories, local_priors, timing)
        if score < current_score - cfg.monotonic_eps:
            if best is None or score < best[4]:
                best = (sc, metrics, dir_loss, rel_loss, score, cand)
        _ = proxy

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
) -> Tuple[Dict[str, Any], Dict[str, Any], float, float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    cur_scene = _deepcopy_scene(scene)
    categories = _build_category_map(cur_scene)
    if fixed_priors:
        priors = list(initial_priors or [])
    else:
        det_priors = _build_deterministic_relation_priors(cur_scene, categories)
        if vlm_priors:
            priors = _merge_relation_priors(cur_scene, categories, det_priors, vlm_priors, cfg.vlm_prior_weight_scale) if cfg.merge_vlm_with_deterministic else list(vlm_priors)
        else:
            priors = det_priors
    cur_score, cur_metrics, cur_dir, cur_rel = _score_scene_full(cur_scene, categories, priors, timing)
    actions: List[Dict[str, Any]] = []

    edit_by_idx: Dict[int, ObjectEdit] = {}
    for edit in edits:
        idx = _find_obj_idx_for_edit(cur_scene, edit)
        if idx is not None and idx not in edit_by_idx:
            edit_by_idx[idx] = edit

    for round_idx in range(cfg.max_rounds):
        improved = False
        categories = _build_category_map(cur_scene)
        if fixed_priors:
            priors = list(initial_priors or [])
        else:
            det_priors = _build_deterministic_relation_priors(cur_scene, categories)
            if vlm_priors:
                priors = _merge_relation_priors(cur_scene, categories, det_priors, vlm_priors, cfg.vlm_prior_weight_scale) if cfg.merge_vlm_with_deterministic else list(vlm_priors)
            else:
                priors = det_priors
        order = _prioritized_object_indices(cur_scene, categories, edits, cur_metrics, cfg)

        for idx in order:
            best = _evaluate_best_local_move(
                cur_scene,
                categories,
                priors,
                idx,
                edit_by_idx.get(idx),
                cur_score,
                cfg,
                fixed_priors=fixed_priors,
                timing=timing,
            )
            if best is None:
                continue
            next_scene, next_metrics, next_dir, next_rel, next_score, cand = best
            actions.append(
                {
                    "round": round_idx,
                    "obj_idx": idx,
                    "kind": cand.get("kind"),
                    "dx": cand.get("dx", 0.0),
                    "dz": cand.get("dz", 0.0),
                    "dyaw": cand.get("dyaw", 0.0),
                    "yaw_abs": cand.get("yaw_abs"),
                    "before_score": cur_score,
                    "after_score": next_score,
                }
            )
            cur_scene = next_scene
            cur_metrics = next_metrics
            cur_dir = next_dir
            cur_rel = next_rel
            cur_score = next_score
            improved = True

        if not improved:
            break

    categories = _build_category_map(cur_scene)
    priors = list(initial_priors or []) if fixed_priors else _build_deterministic_relation_priors(cur_scene, categories)
    _post_refine_functional_layout(cur_scene, categories, priors)
    categories = _build_category_map(cur_scene)
    if fixed_priors:
        priors = list(initial_priors or [])
    else:
        det_priors = _build_deterministic_relation_priors(cur_scene, categories)
        if vlm_priors:
            priors = _merge_relation_priors(cur_scene, categories, det_priors, vlm_priors, cfg.vlm_prior_weight_scale) if cfg.merge_vlm_with_deterministic else list(vlm_priors)
        else:
            priors = det_priors
    cur_score, cur_metrics, cur_dir, cur_rel = _score_scene_full(cur_scene, categories, priors, timing)
    return cur_scene, cur_metrics, cur_dir, cur_rel, actions, priors


# ------------------------------
# main loop
# ------------------------------

def optimize_scene_refactored(
    *,
    scene: Dict[str, Any],
    out_root: Path,
    respace: ReSpace,
    generator: GPTVLMovePromptGeneratorV5,
    extra_hints_text: str,
    cfg: Config,
) -> Dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    step_runtime_records: List[Dict[str, Any]] = []
    overall_t0 = _now()
    current_scene = _deepcopy_scene(scene)
    timing_stats = TimingStats()
    relation_priors_cache: Optional[List[Dict[str, Any]]] = None
    relation_priors_source = "deterministic"
    initial_categories = _build_category_map(current_scene)
    frozen_deterministic_priors = _build_deterministic_relation_priors(current_scene, initial_categories)
    steps_after_valid_pbl = 0

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

        categories = _build_category_map(current_scene)
        deterministic_priors = list(frozen_deterministic_priors) if cfg.freeze_deterministic_priors else _build_deterministic_relation_priors(current_scene, categories)

        if cfg.use_vlm_relation_priors:
            need_refresh = relation_priors_cache is None or cfg.refresh_vlm_relation_priors_every_step
            if need_refresh:
                t_rel0 = _now()
                generated = _safe_generate_relation_priors(
                    generator,
                    diag_path,
                    top_path,
                    current_scene,
                    extra_hints_text,
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
                    print(f"[step {step:02d}] built {len(relation_priors_cache)} VLM relation priors ({rel_gen_sec:.2f}s)")
                else:
                    relation_priors_cache = None
                    relation_priors_source = "deterministic_fallback"
                    print(f"[step {step:02d}] VLM relation priors unavailable, fallback to deterministic")
            current_priors = _merge_relation_priors(current_scene, categories, deterministic_priors, relation_priors_cache, cfg.vlm_prior_weight_scale) if cfg.merge_vlm_with_deterministic else list(relation_priors_cache or deterministic_priors)
        else:
            current_priors = deterministic_priors
            relation_priors_cache = None
            relation_priors_source = "deterministic"

        score_before, metrics_before, dir_before, rel_before = _score_scene_full(current_scene, categories, current_priors, timing_stats)
        pbl_before = _get_float_metric(metrics_before, "total_pbl_loss")

        _write_json(step_dir / "relation_priors.json", current_priors)
        _write_json(step_dir / "relation_priors_meta.json", {"source": relation_priors_source, "count": len(current_priors)})
        print(f"[step {step:02d}] relation priors source={relation_priors_source} count={len(current_priors)}")
        print(
            f"[step {step:02d}] pbl={pbl_before:.6f} oob={_get_float_metric(metrics_before, 'total_oob_loss'):.6f} "
            f"mbl={_get_float_metric(metrics_before, 'total_mbl_loss'):.6f} dir={dir_before:.4f} rel={rel_before:.4f} "
            f"score={score_before:.6f} valid_pbl={pbl_before <= 1e-8}"
        )

        if step > 0 and cfg.stop_when_valid_pbl and pbl_before <= 1e-8:
            if steps_after_valid_pbl >= cfg.max_steps_after_valid_pbl:
                print(f"[step {step:02d}] early stop: valid PBL already preserved for enough cleanup steps")
                break
            if rel_before < 0.35 and dir_before < 0.15:
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
            work_cfg.step_xy = min(cfg.step_xy, 0.16)
            work_cfg.step_yaw = min(cfg.step_yaw, 10.0)
            t_opt0 = _now()
            optimized_scene, metrics_after, dir_after, rel_after, actions, priors_after = _optimize_after_prompt(
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
                extra_hints_text,
                cfg.move_prompt_retries,
                cfg.move_prompt_temperature,
                cfg.move_prompt_max_tokens,
                step_dir,
            )
            prompt_sec = _now() - t_prompt0
            timing_stats.vlm_sec += prompt_sec
            _write_text(step_dir / "move_prompt.txt", prompt_result.move_prompt)
            print(f"[step {step:02d}] move_prompt generated ({len(prompt_result.move_prompt)} chars), temp={cfg.move_prompt_temperature}")

            t_parse0 = _now()
            parse_result = parse_move_prompt(prompt_result.move_prompt)
            parse_sec = _now() - t_parse0
            _write_json(step_dir / "move_prompt_parse.json", {
                "room_name": parse_result.room_name,
                "header_line": parse_result.header_line,
                "parse_warnings": parse_result.parse_warnings,
                "num_edits": len(parse_result.edits),
                "edits": [e.__dict__ for e in parse_result.edits],
            })

            scene_after_prompt = _deepcopy_scene(current_scene)
            scene_after_prompt, applied_count, changes = apply_edits_to_scene(scene_after_prompt, parse_result.edits)
            _write_json(step_dir / "applied_edits.json", {"applied_count": applied_count, "changes": changes})

            t_opt0 = _now()
            optimized_scene, metrics_after, dir_after, rel_after, actions, priors_after = _optimize_after_prompt(
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
        score_after = _W_PBL * _get_float_metric(metrics_after, "total_pbl_loss") + _W_DIR * dir_after + _W_REL * rel_after

        t_post0 = _now()
        _write_json(step_dir / "optimizer_actions.json", actions)
        _write_json(step_dir / "scene_after.json", optimized_scene)
        _write_json(step_dir / "relation_priors_after.json", priors_after)
        _write_json(step_dir / "relation_priors_after_meta.json", {"source": relation_priors_source, "count": len(priors_after)})
        post_sec = _now() - t_post0

        total_sec = _now() - step_t0
        print(
            f"[step {step:02d}] time total={total_sec:.2f}s (prompt={prompt_sec:.2f}s, parse={parse_sec:.2f}s, "
            f"opt={opt_sec:.2f}s, post={post_sec:.2f}s) applied={applied_count} "
            f"pbl {pbl_before:.6f}->{_get_float_metric(metrics_after, 'total_pbl_loss'):.6f} "
            f"dir {dir_before:.4f}->{dir_after:.4f} rel {rel_before:.4f}->{rel_after:.4f} "
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
            elif score_after < score_before - cfg.min_score_improve_after_valid_pbl:
                accepted = True
            else:
                reject_reason = "improvement_too_small"
        else:
            pbl_improved = pbl_after < pbl_before - cfg.monotonic_eps
            score_guard = score_after <= score_before + cfg.max_score_increase_prevalid
            dir_guard = dir_after <= dir_before + cfg.max_dir_increase_prevalid
            rel_guard = rel_after <= rel_before + cfg.max_rel_increase_prevalid

            if pbl_improved and score_guard and dir_guard and rel_guard:
                accepted = True
            elif score_after < score_before - cfg.monotonic_eps and pbl_after <= pbl_before + 0.005:
                accepted = True
            else:
                if not pbl_improved:
                    reject_reason = "pbl_not_better"
                elif not score_guard:
                    reject_reason = "score_worse_too_much"
                elif not dir_guard:
                    reject_reason = "direction_worse_too_much"
                else:
                    reject_reason = "relation_worse_too_much"

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
            }
        )

        if cfg.stop_when_valid_pbl and accepted and _get_float_metric(metrics_after, "total_pbl_loss") <= 1e-8 and dir_after < 0.15 and rel_after < 0.35:
            break

    if cfg.render_final:
        final_dir = out_root / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        t_render0 = _now()
        respace.render_scene_frame(current_scene, filename="final", pth_viz_output=final_dir)
        render_annotated_top_view(current_scene, "final", final_dir, resolution=(1024, 1024), show_assets=True, font_size=14)
        timing_stats.render_sec += _now() - t_render0
        _write_json(final_dir / "scene.json", current_scene)

    cats = _build_category_map(current_scene)
    deterministic_priors = list(frozen_deterministic_priors) if cfg.freeze_deterministic_priors else _build_deterministic_relation_priors(current_scene, cats)
    if cfg.use_vlm_relation_priors and relation_priors_cache:
        priors = _merge_relation_priors(current_scene, cats, deterministic_priors, relation_priors_cache, cfg.vlm_prior_weight_scale) if cfg.merge_vlm_with_deterministic else relation_priors_cache
    else:
        priors = deterministic_priors
        if not cfg.use_vlm_relation_priors:
            relation_priors_source = "deterministic"
    final_score, final_metrics, final_dir_loss, final_rel_loss = _score_scene_full(current_scene, cats, priors, timing_stats)
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
    
    # 从环境变量读取可配置的 prompt（保持默认值=原硬编码）
    room_prompt = os.getenv(
        "ROOM_PROMPT",
    ).strip()

    updated_scene, is_success = respace.handle_prompt(room_prompt, scene)
    _ = is_success  # 保持原逻辑不变（如果你后面要用 is_success 可去掉这行）
    
    out_root = Path(os.getenv("OUT_DIR", "./evaluate/gpt_image_describe_refactored")).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        max_steps=int(os.getenv("MAX_STEPS", "3")),
        max_rounds=int(os.getenv("MAX_ROUNDS", "2")),
        max_objects_per_round=int(os.getenv("MAX_OBJECTS_PER_ROUND", "6")),
        proxy_topk=int(os.getenv("PROXY_TOPK", "2")),
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
        freeze_deterministic_priors=os.getenv("FREEZE_DETERMINISTIC_PRIORS", "1") not in ("0", "false", "False", ""),
        local_repair_passes=int(os.getenv("LOCAL_REPAIR_PASSES", "1")),
        full_repair_after_refine_passes=int(os.getenv("FULL_REPAIR_AFTER_REFINE_PASSES", "1")),
        stop_when_valid_pbl=os.getenv("STOP_WHEN_VALID_PBL", "1") not in ("0", "false", "False", ""),
        cleanup_only_after_valid_pbl=os.getenv("CLEANUP_ONLY_AFTER_VALID_PBL", "1") not in ("0", "false", "False", ""),
        max_steps_after_valid_pbl=int(os.getenv("MAX_STEPS_AFTER_VALID_PBL", "1")),
        max_objects_after_valid_pbl=int(os.getenv("MAX_OBJECTS_AFTER_VALID_PBL", "3")),
        max_rounds_after_valid_pbl=int(os.getenv("MAX_ROUNDS_AFTER_VALID_PBL", "1")),
        min_score_improve_after_valid_pbl=float(os.getenv("MIN_SCORE_IMPROVE_AFTER_VALID_PBL", "0.03")),
        max_dir_increase_after_valid_pbl=float(os.getenv("MAX_DIR_INCREASE_AFTER_VALID_PBL", "0.03")),
        max_rel_increase_after_valid_pbl=float(os.getenv("MAX_REL_INCREASE_AFTER_VALID_PBL", "0.10")),
        max_score_increase_prevalid=float(os.getenv("MAX_SCORE_INCREASE_PREVALID", "0.05")),
        max_dir_increase_prevalid=float(os.getenv("MAX_DIR_INCREASE_PREVALID", "0.10")),
        max_rel_increase_prevalid=float(os.getenv("MAX_REL_INCREASE_PREVALID", "0.50")),
        render_final=os.getenv("RENDER_FINAL", "1") not in ("0", "false", "False", ""),
        monotonic_eps=float(os.getenv("MONOTONIC_EPS", "1e-12")),
        step_xy=float(os.getenv("STEP_XY", "0.22")),
        step_yaw=float(os.getenv("STEP_YAW", "15.0")),
        anchor_lock_pbl_threshold=float(os.getenv("ANCHOR_LOCK_PBL_THRESHOLD", "0.08")),
        role_refine_blend=float(os.getenv("ROLE_REFINE_BLEND", "0.72")),
    )

    extra_hints_text = (
        "GLOBAL SAFETY CONSTRAINTS:\n"
        "1) All objects must stay fully inside the room.\n"
        "2) Avoid overlaps; keep small but visible clearance.\n"
        "3) Prioritize OOB and collision fixes before aesthetics.\n"
        "4) Chairs should face their functional anchors.\n"
        "5) Bedside furniture should remain close to the bed sides."
    )

    model = os.getenv("MOVE_PROMPT_MODEL", os.getenv("YUNWU_AI_MODEL", "gpt-4o"))
    generator = GPTVLMovePromptGeneratorV5(
        model=model,
        api_base=os.getenv("YUNWU_AI_API_BASE"),
        api_key=os.getenv("YUNWU_AI_API_KEY"),
        timeout_s=float(os.getenv("MOVE_PROMPT_TIMEOUT_S", "120")),
    )

    summary = optimize_scene_refactored(
        scene=updated_scene,
        out_root=out_root,
        respace=ReSpace(),
        generator=generator,
        extra_hints_text=extra_hints_text,
        cfg=cfg,
    )
    print("\n=== Done ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
