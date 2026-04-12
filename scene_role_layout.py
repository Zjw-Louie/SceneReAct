from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

REL_TYPES = {
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
    "dresser": "cabinet",
    "desk": "desk",
    "dressing table": "vanity",
    "writing desk": "desk",
    "computer desk": "desk",
    "study desk": "desk",
    "vanity": "vanity",
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
    "sink": "sink",
    "wash basin": "sink",
    "counter": "counter",
    "kitchen island": "counter",
    "stove": "stove",
    "cooktop": "stove",
    "oven": "oven",
    "range hood": "hood",
    "refrigerator": "refrigerator",
    "fridge": "refrigerator",
    "toilet": "toilet",
    "bathtub": "bathtub",
    "tub": "bathtub",
    "shower": "shower",
    "washing machine": "washing machine",
    "washer": "washing machine",
    "dryer": "dryer",
    "laundry basket": "laundry basket",
    "bench": "bench",
}

KEYWORD_CATEGORIES = (
    "bed",
    "nightstand",
    "wardrobe",
    "desk",
    "vanity",
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
    "sink",
    "counter",
    "stove",
    "oven",
    "refrigerator",
    "toilet",
    "bathtub",
    "shower",
    "washing machine",
    "dryer",
)

FUNCTION_BY_CATEGORY = {
    "bed": "sleep",
    "nightstand": "sleep",
    "desk": "work",
    "vanity": "wash",
    "chair": "seating",
    "table": "eat",
    "coffee table": "relax",
    "sofa": "relax",
    "tv stand": "relax",
    "cabinet": "store",
    "wardrobe": "store",
    "shelf": "store",
    "mirror": "wash",
    "sink": "wash",
    "toilet": "wash",
    "bathtub": "wash",
    "shower": "wash",
    "counter": "cook",
    "stove": "cook",
    "oven": "cook",
    "hood": "cook",
    "refrigerator": "cook",
    "washing machine": "laundry",
    "dryer": "laundry",
    "laundry basket": "laundry",
    "lamp": "decor",
    "plant": "decor",
    "rug": "decor",
    "curtain": "decor",
    "bench": "seating",
}

SMALL_ACCESSORY_CATEGORIES = {
    "nightstand",
    "lamp",
    "mirror",
    "plant",
    "rug",
    "laundry basket",
}

WALL_AFFINE_CATEGORIES = {
    "bed",
    "wardrobe",
    "cabinet",
    "shelf",
    "tv stand",
    "console table",
    "vanity",
    "sink",
    "toilet",
    "bathtub",
    "shower",
    "counter",
    "stove",
    "oven",
    "refrigerator",
    "washing machine",
    "dryer",
    "mirror",
}

PARALLEL_WALL_CATEGORIES = {
    "table",
    "counter",
    "console table",
    "tv stand",
    "cabinet",
    "desk",
    "vanity",
}

INTERACTIVE_CATEGORIES = {
    "desk",
    "table",
    "vanity",
    "sink",
    "toilet",
    "counter",
    "stove",
    "oven",
    "washing machine",
    "dryer",
    "wardrobe",
    "cabinet",
    "refrigerator",
    "bed",
    "sofa",
}

SUPPORT_SURFACE_CATEGORIES = {
    "desk",
    "table",
    "nightstand",
    "cabinet",
    "console table",
    "tv stand",
    "vanity",
    "counter",
}

SEATING_COMPATIBLE_ANCHORS = {
    "desk",
    "table",
    "vanity",
    "counter",
    "sofa",
    "bench",
}

DEFAULT_ACCESS_WIDTH = {
    "bed": 0.8,
    "desk": 0.75,
    "table": 0.85,
    "vanity": 0.7,
    "sink": 0.65,
    "toilet": 0.6,
    "counter": 0.8,
    "stove": 0.75,
    "oven": 0.7,
    "washing machine": 0.7,
    "dryer": 0.7,
    "wardrobe": 0.75,
    "cabinet": 0.65,
    "refrigerator": 0.8,
    "sofa": 0.7,
}

DEFAULT_ZONE_BY_ROOM = {
    "bedroom": {"sleep", "work", "relax", "store"},
    "living room": {"relax", "eat", "store"},
    "kitchen": {"cook", "eat", "store"},
    "bathroom": {"wash", "store"},
    "laundry room": {"laundry", "store"},
}


@dataclass
class RoleGraph:
    categories: List[str]
    role_by_idx: Dict[int, str]
    function_by_idx: Dict[int, str]
    zone_by_idx: Dict[int, str]
    dominant_indices: Set[int]
    accessory_indices: Set[int]
    wall_affine_indices: Set[int]
    parallel_wall_indices: Set[int]
    interactive_indices: Set[int]
    support_surface_indices: Set[int]
    open_front_indices: Set[int]
    seating_indices: Set[int]
    accessory_to_anchor: Dict[int, int]
    anchor_capacity: Dict[int, int]
    notes: List[str] = field(default_factory=list)


@dataclass
class FunctionalViolations:
    total: float
    violations: List[Dict[str, Any]]


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


def normalize_category(category: str) -> str:
    category = (category or "").strip().lower()
    if category in NORMALIZE_CATEGORY_MAP:
        return NORMALIZE_CATEGORY_MAP[category]
    for key, value in NORMALIZE_CATEGORY_MAP.items():
        if key in category:
            return value
    for keyword in KEYWORD_CATEGORIES:
        if keyword in category:
            return keyword
    return category or "unknown"


def _category_from_desc(desc: str) -> str:
    desc = (desc or "").lower()
    for key, value in NORMALIZE_CATEGORY_MAP.items():
        if key in desc:
            return value
    return "unknown"


def get_obj_category(obj: Dict[str, Any]) -> str:
    jid = obj.get("sampled_asset_jid") or obj.get("jid") or ""
    info = _load_model_info()

    for key in [jid, jid.split("-(")[0] if "-(" in jid else jid]:
        if key in info:
            category = (info[key].get("category") or info[key].get("super-category") or "").strip().lower()
            if category:
                return normalize_category(category)

    base = jid.split("-(")[0] if "-(" in jid else jid
    if "_" in base:
        prefix = base.rsplit("_", 1)[0]
        if prefix in info:
            category = (info[prefix].get("category") or info[prefix].get("super-category") or "").strip().lower()
            if category:
                return normalize_category(category)

    desc = obj.get("desc") or obj.get("description") or obj.get("style_description") or ""
    return _category_from_desc(desc)


def obj_size_xz(obj: Dict[str, Any]) -> Tuple[float, float]:
    size = obj.get("size", [1.0, 1.0, 1.0])
    sx = float(size[0]) if len(size) > 0 else 1.0
    sz = float(size[2]) if len(size) > 2 else 1.0
    return sx, sz


def obj_diag_size_xz(obj: Dict[str, Any]) -> float:
    sx, sz = obj_size_xz(obj)
    return math.hypot(sx, sz)


def xz_dist(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    pa = a.get("pos", [0.0, 0.0, 0.0])
    pb = b.get("pos", [0.0, 0.0, 0.0])
    return math.hypot(pa[0] - pb[0], pa[2] - pb[2])


def build_category_map(scene: Dict[str, Any]) -> List[str]:
    initial = [get_obj_category(obj) for obj in scene.get("objects", [])]
    return refine_categories_contextual(scene, initial)


def room_center_xz(scene: Dict[str, Any]) -> Tuple[float, float]:
    pts = [(float(p[0]), float(p[2])) for p in scene.get("bounds_bottom", []) if isinstance(p, list) and len(p) >= 3]
    if not pts:
        return 0.0, 0.0
    return sum(x for x, _ in pts) / len(pts), sum(z for _, z in pts) / len(pts)


def distance_to_nearest_wall_xz(scene: Dict[str, Any], pos: Sequence[float]) -> float:
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


def refine_categories_contextual(scene: Dict[str, Any], categories: List[str]) -> List[str]:
    objs = scene.get("objects", [])
    refined = list(categories)
    bed_indices = [i for i, cat in enumerate(categories) if cat == "bed"]
    sink_indices = [i for i, cat in enumerate(categories) if cat == "sink"]
    vanity_indices = [i for i, cat in enumerate(categories) if cat == "vanity"]

    for i, cat in enumerate(categories):
        obj = objs[i]
        sx, sz = obj_size_xz(obj)
        max_side = max(sx, sz)
        min_side = min(sx, sz)
        area = sx * sz
        wall_d = distance_to_nearest_wall_xz(scene, obj.get("pos", [0.0, 0.0, 0.0]))

        small_table_like = cat in {"cabinet", "shelf", "desk", "table", "unknown"} and max_side <= 0.8
        if small_table_like and bed_indices:
            nearest_bed = min(xz_dist(obj, objs[b]) for b in bed_indices)
            if nearest_bed <= 1.2:
                refined[i] = "nightstand"
                continue

        if cat in {"table", "desk"} and wall_d < 0.25 and max_side <= 1.4 and min_side <= 0.55:
            refined[i] = "console table"
            continue

        if cat in {"cabinet", "table", "unknown"} and wall_d < 0.35 and max_side <= 1.4 and area <= 1.5:
            near_sink = min((xz_dist(obj, objs[j]) for j in sink_indices), default=999.0)
            if near_sink <= 1.3:
                refined[i] = "vanity"
                continue

        if cat == "mirror" and not vanity_indices and sink_indices:
            near_sink = min((xz_dist(obj, objs[j]) for j in sink_indices), default=999.0)
            if near_sink <= 1.6:
                refined[i] = "mirror"

    return refined


def infer_function(category: str, room_type: str = "") -> str:
    category = normalize_category(category)
    if category in FUNCTION_BY_CATEGORY:
        return FUNCTION_BY_CATEGORY[category]
    room_type = (room_type or "").lower()
    if "bath" in room_type:
        return "wash"
    if "kitchen" in room_type:
        return "cook"
    if "bed" in room_type:
        return "sleep"
    if "living" in room_type:
        return "relax"
    return "general"


def _infer_role(category: str, obj: Dict[str, Any]) -> str:
    category = normalize_category(category)
    sx, sz = obj_size_xz(obj)
    area = sx * sz
    if category in {"chair", "bench"}:
        return "seating"
    if category in SMALL_ACCESSORY_CATEGORIES:
        return "accessory"
    if category in {"mirror", "rug", "plant", "lamp", "curtain"}:
        return "decor"
    if category in SUPPORT_SURFACE_CATEGORIES:
        return "dominant_anchor"
    if category in {
        "bed",
        "wardrobe",
        "cabinet",
        "shelf",
        "sofa",
        "tv stand",
        "sink",
        "counter",
        "stove",
        "oven",
        "refrigerator",
        "toilet",
        "bathtub",
        "shower",
        "washing machine",
        "dryer",
    }:
        return "dominant_anchor"
    if max(sx, sz) >= 1.1 or area >= 0.9:
        return "dominant_anchor"
    return "accessory"


def _anchor_capacity_for_category(category: str) -> int:
    category = normalize_category(category)
    if category in {"desk", "vanity", "sink", "toilet", "stove", "oven", "washing machine", "dryer", "wardrobe"}:
        return 1
    if category in {"bed", "sofa"}:
        return 2
    if category in {"table", "counter", "cabinet"}:
        return 3
    return 2


def _compatibility_penalty(acc_cat: str, anchor_cat: str, acc_fn: str, anchor_fn: str) -> float:
    acc_cat = normalize_category(acc_cat)
    anchor_cat = normalize_category(anchor_cat)
    penalty = 0.0
    if acc_cat == "chair":
        if anchor_cat not in SEATING_COMPATIBLE_ANCHORS and anchor_fn not in {"work", "eat", "relax", "wash", "cook"}:
            return 4.0
        if anchor_cat == "sofa":
            penalty += 0.25
        if anchor_cat == "counter":
            penalty += 0.15
    elif acc_cat == "nightstand":
        if anchor_cat != "bed":
            return 5.0
    elif acc_cat == "mirror":
        if anchor_cat not in {"vanity", "sink", "desk", "cabinet"}:
            penalty += 1.2
    elif acc_cat == "rug":
        if anchor_cat not in {"bed", "sofa", "table"}:
            penalty += 0.8
    elif acc_cat == "lamp":
        if anchor_cat not in SUPPORT_SURFACE_CATEGORIES and anchor_cat not in {"bed", "sofa", "desk"}:
            penalty += 0.8
    elif acc_cat == "plant":
        penalty += 0.2
    elif acc_cat == "laundry basket":
        if anchor_fn != "laundry":
            penalty += 1.0

    if acc_fn not in {"decor", "general", "seating"} and anchor_fn != acc_fn:
        penalty += 0.8
    return penalty


def _preferred_accessory_distance(acc_cat: str, anchor_cat: str, acc_obj: Dict[str, Any], anchor_obj: Dict[str, Any]) -> float:
    base = 0.32 * (obj_diag_size_xz(acc_obj) + obj_diag_size_xz(anchor_obj)) + 0.18
    acc_cat = normalize_category(acc_cat)
    anchor_cat = normalize_category(anchor_cat)
    if acc_cat == "chair":
        return max(0.58, base + 0.12)
    if acc_cat == "nightstand":
        return max(0.48, base - 0.05)
    if acc_cat == "mirror":
        return max(0.3, base - 0.1)
    if acc_cat == "rug":
        return max(0.0, base)
    if acc_cat == "lamp":
        return max(0.22, base - 0.1)
    if anchor_cat in {"bed", "sofa"}:
        return max(0.5, base)
    return max(0.35, base)


def _assign_accessories_to_anchors(scene: Dict[str, Any], categories: List[str], role_by_idx: Dict[int, str], function_by_idx: Dict[int, str]) -> Tuple[Dict[int, int], Dict[int, int]]:
    objs = scene.get("objects", [])
    dominant_indices = [i for i, role in role_by_idx.items() if role == "dominant_anchor"]
    accessory_indices = [i for i, role in role_by_idx.items() if role in {"accessory", "seating", "decor"}]
    capacities = {i: _anchor_capacity_for_category(categories[i]) for i in dominant_indices}
    used = {i: 0 for i in dominant_indices}
    assignment: Dict[int, int] = {}

    scored_pairs: List[Tuple[float, int, int]] = []
    for acc_idx in accessory_indices:
        acc_obj = objs[acc_idx]
        acc_cat = categories[acc_idx]
        acc_fn = function_by_idx[acc_idx]
        for anchor_idx in dominant_indices:
            anchor_obj = objs[anchor_idx]
            anchor_cat = categories[anchor_idx]
            anchor_fn = function_by_idx[anchor_idx]
            compat = _compatibility_penalty(acc_cat, anchor_cat, acc_fn, anchor_fn)
            d = xz_dist(acc_obj, anchor_obj)
            target = _preferred_accessory_distance(acc_cat, anchor_cat, acc_obj, anchor_obj)
            score = compat + abs(d - target)
            if acc_cat == "mirror" and anchor_idx in used:
                score += 0.1 * used[anchor_idx]
            scored_pairs.append((score, acc_idx, anchor_idx))

    scored_pairs.sort(key=lambda x: x[0])
    for score, acc_idx, anchor_idx in scored_pairs:
        if acc_idx in assignment:
            continue
        if used[anchor_idx] >= capacities[anchor_idx]:
            continue
        if score > 5.5:
            continue
        assignment[acc_idx] = anchor_idx
        used[anchor_idx] += 1
    return assignment, capacities


def infer_role_graph(scene: Dict[str, Any], categories: Optional[List[str]] = None) -> RoleGraph:
    categories = list(categories or build_category_map(scene))
    room_type = str(scene.get("room_type", "")).lower()
    objs = scene.get("objects", [])
    role_by_idx: Dict[int, str] = {}
    function_by_idx: Dict[int, str] = {}
    zone_by_idx: Dict[int, str] = {}
    dominant_indices: Set[int] = set()
    accessory_indices: Set[int] = set()
    wall_affine_indices: Set[int] = set()
    parallel_wall_indices: Set[int] = set()
    interactive_indices: Set[int] = set()
    support_surface_indices: Set[int] = set()
    open_front_indices: Set[int] = set()
    seating_indices: Set[int] = set()
    notes: List[str] = []

    for i, obj in enumerate(objs):
        cat = normalize_category(categories[i])
        categories[i] = cat
        role = _infer_role(cat, obj)
        func = infer_function(cat, room_type=room_type)
        role_by_idx[i] = role
        function_by_idx[i] = func
        zone_by_idx[i] = func

        if role == "dominant_anchor":
            dominant_indices.add(i)
        else:
            accessory_indices.add(i)
        if cat in WALL_AFFINE_CATEGORIES:
            wall_affine_indices.add(i)
        if cat in PARALLEL_WALL_CATEGORIES:
            parallel_wall_indices.add(i)
        if cat in INTERACTIVE_CATEGORIES:
            interactive_indices.add(i)
            open_front_indices.add(i)
        if cat in SUPPORT_SURFACE_CATEGORIES:
            support_surface_indices.add(i)
        if cat in {"chair", "bench"}:
            seating_indices.add(i)

    accessory_to_anchor, anchor_capacity = _assign_accessories_to_anchors(scene, categories, role_by_idx, function_by_idx)
    for acc_idx, anchor_idx in accessory_to_anchor.items():
        zone_by_idx[acc_idx] = zone_by_idx.get(anchor_idx, function_by_idx[acc_idx])
        if function_by_idx[acc_idx] in {"seating", "decor", "general"}:
            function_by_idx[acc_idx] = function_by_idx.get(anchor_idx, function_by_idx[acc_idx])

    valid_room_zones = DEFAULT_ZONE_BY_ROOM.get(room_type, set())
    if valid_room_zones:
        for idx, z in list(zone_by_idx.items()):
            if z not in valid_room_zones and z not in {"decor", "general", "store", "seating"}:
                notes.append(f"zone_override:{idx}:{z}->general")
                zone_by_idx[idx] = "general"

    return RoleGraph(
        categories=categories,
        role_by_idx=role_by_idx,
        function_by_idx=function_by_idx,
        zone_by_idx=zone_by_idx,
        dominant_indices=dominant_indices,
        accessory_indices=accessory_indices,
        wall_affine_indices=wall_affine_indices,
        parallel_wall_indices=parallel_wall_indices,
        interactive_indices=interactive_indices,
        support_surface_indices=support_surface_indices,
        open_front_indices=open_front_indices,
        seating_indices=seating_indices,
        accessory_to_anchor=accessory_to_anchor,
        anchor_capacity=anchor_capacity,
        notes=notes,
    )


def major_anchor_indices(role_graph: RoleGraph) -> Set[int]:
    return set(role_graph.dominant_indices)


def build_role_based_relation_priors(scene: Dict[str, Any], role_graph: RoleGraph) -> List[Dict[str, Any]]:
    objs = scene.get("objects", [])
    priors: List[Dict[str, Any]] = []

    for idx in sorted(role_graph.wall_affine_indices):
        cat = role_graph.categories[idx]
        rel_type = "parallel" if idx in role_graph.parallel_wall_indices and cat != "bed" else "against_wall"
        priors.append({"src_idx": idx, "type": rel_type, "confidence": 0.85, "weight": 0.9})

    for acc_idx, anchor_idx in role_graph.accessory_to_anchor.items():
        acc_cat = role_graph.categories[acc_idx]
        anchor_cat = role_graph.categories[anchor_idx]
        priors.append({"src_idx": acc_idx, "tgt_idx": anchor_idx, "type": "near", "confidence": 0.9, "weight": 1.0})
        if acc_cat == "chair":
            rel_type = "centered_with" if anchor_cat in {"desk", "vanity", "counter", "sink"} else "facing"
            priors.append({"src_idx": acc_idx, "tgt_idx": anchor_idx, "type": rel_type, "confidence": 0.86, "weight": 1.0})
        elif acc_cat in {"nightstand", "lamp"} and anchor_cat == "bed":
            priors.append({"src_idx": acc_idx, "tgt_idx": anchor_idx, "type": "side_of", "confidence": 0.82, "weight": 0.9})
        elif acc_cat == "rug" and anchor_cat in {"bed", "sofa", "table"}:
            priors.append({"src_idx": acc_idx, "tgt_idx": anchor_idx, "type": "near", "confidence": 0.78, "weight": 0.8})

    dominant = sorted(role_graph.dominant_indices)
    for i, src_idx in enumerate(dominant):
        for tgt_idx in dominant[i + 1 :]:
            if role_graph.zone_by_idx.get(src_idx) == role_graph.zone_by_idx.get(tgt_idx):
                d = xz_dist(objs[src_idx], objs[tgt_idx])
                if d <= 3.5:
                    priors.append({"src_idx": src_idx, "tgt_idx": tgt_idx, "type": "distance_band", "confidence": 0.62, "weight": 0.55})

    unique: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for item in priors:
        key = (item.get("src_idx"), item.get("tgt_idx"), item.get("type"))
        score = float(item.get("confidence", 0.0)) * float(item.get("weight", 1.0))
        old = unique.get(key)
        old_score = float(old.get("confidence", 0.0)) * float(old.get("weight", 1.0)) if old else -1.0
        if old is None or score > old_score:
            unique[key] = item
    return list(unique.values())


def optimization_stage_order(
    scene: Dict[str, Any],
    role_graph: RoleGraph,
    edited_indices: Sequence[int],
    hotspot_idx: Optional[int],
    max_objects: int,
    lock_major: bool,
) -> List[int]:
    order: List[int] = []
    if hotspot_idx is not None:
        order.append(hotspot_idx)
    order.extend(i for i in edited_indices if i not in order)

    dominant = sorted(role_graph.dominant_indices)
    accessories = sorted(role_graph.accessory_indices)
    if lock_major:
        dominant = [i for i in dominant if i == hotspot_idx or i in edited_indices]

    def _dominant_priority(idx: int) -> Tuple[float, float]:
        cat = role_graph.categories[idx]
        wall_d = distance_to_nearest_wall_xz(scene, scene["objects"][idx].get("pos", [0.0, 0.0, 0.0]))
        cat_bonus = 0.0 if cat in {"bed", "desk", "table", "sofa", "counter", "sink", "toilet", "washing machine"} else 0.3
        return (cat_bonus, wall_d)

    def _accessory_priority(idx: int) -> Tuple[float, float]:
        anchor_idx = role_graph.accessory_to_anchor.get(idx)
        dist = xz_dist(scene["objects"][idx], scene["objects"][anchor_idx]) if anchor_idx is not None else 999.0
        return (0.0 if anchor_idx is not None else 1.0, dist)

    for idx in sorted(dominant, key=_dominant_priority):
        if idx not in order:
            order.append(idx)
    for idx in sorted(accessories, key=_accessory_priority):
        if idx not in order:
            order.append(idx)
    return order[:max_objects]


def choose_open_side_for_anchor(scene: Dict[str, Any], anchor: Dict[str, Any], anchor_cat: str) -> Tuple[float, float]:
    pos = anchor.get("pos", [0.0, 0.0, 0.0])
    yaw = float(anchor.get("_yaw", 0.0))
    if not math.isfinite(yaw):
        yaw = 0.0
    rad = math.radians(yaw)
    fx, fz = math.sin(rad), math.cos(rad)
    lx, lz = fz, -fx

    if anchor_cat in {"desk", "vanity", "sink", "counter", "stove", "oven"}:
        candidates = [((fx, fz), 0.95), ((lx, lz), 0.12), ((-lx, -lz), 0.12), ((-fx, -fz), -0.25)]
    elif anchor_cat in {"bed", "sofa"}:
        candidates = [((lx, lz), 0.32), ((-lx, -lz), 0.32), ((fx, fz), 0.05), ((-fx, -fz), 0.0)]
    else:
        candidates = [((fx, fz), 0.3), ((-fx, -fz), 0.1), ((lx, lz), 0.1), ((-lx, -lz), 0.1)]

    room_cx, room_cz = room_center_xz(scene)
    best_vec = candidates[0][0]
    best_score = -1e9
    for (vx, vz), bias in candidates:
        sample_x = pos[0] + vx * 0.75
        sample_z = pos[2] + vz * 0.75
        wall_d = distance_to_nearest_wall_xz(scene, [sample_x, 0.0, sample_z])
        center_bonus = 0.18 * ((room_cx - pos[0]) * vx + (room_cz - pos[2]) * vz)
        score = bias + min(1.35, wall_d) + center_bonus
        if score > best_score:
            best_score = score
            best_vec = (vx, vz)
    return best_vec


def target_pose_for_attachment(
    scene: Dict[str, Any],
    objs: List[Dict[str, Any]],
    acc_idx: int,
    anchor_idx: int,
    yaw_from_quaternion,
) -> Tuple[List[float], float]:
    acc_obj = objs[acc_idx]
    anchor_obj = objs[anchor_idx]
    acc_cat = normalize_category(acc_obj.get("_category", ""))
    anchor_cat = normalize_category(anchor_obj.get("_category", ""))
    ax, ay, az = anchor_obj.get("pos", [0.0, 0.0, 0.0])
    ayaw = yaw_from_quaternion(anchor_obj.get("rot", [0.0, 0.0, 0.0, 1.0]))
    anchor_with_yaw = dict(anchor_obj)
    anchor_with_yaw["_yaw"] = ayaw
    vx, vz = choose_open_side_for_anchor(scene, anchor_with_yaw, anchor_cat)
    dist = _preferred_accessory_distance(acc_cat, anchor_cat, acc_obj, anchor_obj)

    if acc_cat in {"nightstand", "lamp"} and anchor_cat == "bed":
        rad = math.radians(ayaw)
        fx, fz = math.sin(rad), math.cos(rad)
        lx, lz = fz, -fx
        apos = acc_obj.get("pos", [0.0, 0.0, 0.0])
        side = -1.0 if ((apos[0] - ax) * lx + (apos[2] - az) * lz) < 0 else 1.0
        vx, vz = side * lx, side * lz
    elif acc_cat == "mirror" and anchor_cat in {"vanity", "sink", "desk", "cabinet"}:
        wall_d = distance_to_nearest_wall_xz(scene, anchor_obj.get("pos", [0.0, 0.0, 0.0]))
        if wall_d < 0.65:
            vx, vz = -vx, -vz
            dist = max(0.4, dist)

    target_pos = [ax + vx * dist, acc_obj.get("pos", [0.0, 0.0, 0.0])[1], az + vz * dist]
    target_yaw = math.degrees(math.atan2(ax - target_pos[0], az - target_pos[2])) % 360.0
    return target_pos, target_yaw


def compute_functional_loss(scene: Dict[str, Any], role_graph: RoleGraph, yaw_from_quaternion) -> FunctionalViolations:
    objs = scene.get("objects", [])
    total = 0.0
    violations: List[Dict[str, Any]] = []

    for acc_idx, anchor_idx in role_graph.accessory_to_anchor.items():
        d = xz_dist(objs[acc_idx], objs[anchor_idx])
        target = _preferred_accessory_distance(role_graph.categories[acc_idx], role_graph.categories[anchor_idx], objs[acc_idx], objs[anchor_idx])
        tol = 0.3 if role_graph.categories[acc_idx] == "chair" else 0.25
        if d > target + tol:
            penalty = d - target - tol
            total += penalty
            violations.append({"kind": "accessory_far", "idx": acc_idx, "anchor_idx": anchor_idx, "penalty": round(penalty, 4)})

    for idx in role_graph.wall_affine_indices:
        cat = role_graph.categories[idx]
        pos = objs[idx].get("pos", [0.0, 0.0, 0.0])
        wall_d = distance_to_nearest_wall_xz(scene, pos)
        limit = 0.42 if cat in {"bed", "sofa"} else 0.35
        if wall_d > limit:
            penalty = (wall_d - limit) * 0.6
            total += penalty
            violations.append({"kind": "wall_affinity", "idx": idx, "penalty": round(penalty, 4)})

    for idx in role_graph.open_front_indices:
        obj = objs[idx]
        cat = role_graph.categories[idx]
        yaw = yaw_from_quaternion(obj.get("rot", [0.0, 0.0, 0.0, 1.0]))
        obj_with_yaw = dict(obj)
        obj_with_yaw["_yaw"] = yaw
        vx, vz = choose_open_side_for_anchor(scene, obj_with_yaw, cat)
        access_len = max(0.75, 0.42 + obj_diag_size_xz(obj) * 0.35)
        access_width = DEFAULT_ACCESS_WIDTH.get(cat, 0.7)
        ox, oz = obj.get("pos", [0.0, 0.0, 0.0])[0], obj.get("pos", [0.0, 0.0, 0.0])[2]
        blocked = 0.0
        for j, other in enumerate(objs):
            if j == idx:
                continue
            px, pz = other.get("pos", [0.0, 0.0, 0.0])[0] - ox, other.get("pos", [0.0, 0.0, 0.0])[2] - oz
            forward = px * vx + pz * vz
            lateral = abs(-px * vz + pz * vx)
            if 0.18 < forward < access_len and lateral < access_width * 0.5:
                blocked += max(0.0, access_len - forward) * 0.35
        if blocked > 0.08:
            total += blocked
            violations.append({"kind": "access_blocked", "idx": idx, "penalty": round(blocked, 4)})

    dominant = sorted(role_graph.dominant_indices)
    for i, src_idx in enumerate(dominant):
        for tgt_idx in dominant[i + 1 :]:
            d = xz_dist(objs[src_idx], objs[tgt_idx])
            min_sep = 0.24 * (obj_diag_size_xz(objs[src_idx]) + obj_diag_size_xz(objs[tgt_idx]))
            same_zone = role_graph.zone_by_idx.get(src_idx) == role_graph.zone_by_idx.get(tgt_idx)
            if not same_zone and d < min_sep + 0.35:
                penalty = (min_sep + 0.35 - d) * 0.65
                total += penalty
                violations.append({"kind": "cross_zone_too_close", "src_idx": src_idx, "tgt_idx": tgt_idx, "penalty": round(penalty, 4)})
            elif same_zone and d > 4.5:
                penalty = (d - 4.5) * 0.12
                total += penalty
                violations.append({"kind": "same_zone_too_far", "src_idx": src_idx, "tgt_idx": tgt_idx, "penalty": round(penalty, 4)})

    return FunctionalViolations(total=total, violations=violations)


def post_refine_role_layout(
    scene: Dict[str, Any],
    role_graph: RoleGraph,
    yaw_from_quaternion,
    quaternion_from_yaw,
    repair_callback,
    blend: float = 0.65,
    full_repair_passes: int = 1,
) -> None:
    objs = scene.get("objects", [])

    for idx in sorted(role_graph.dominant_indices):
        if idx in role_graph.wall_affine_indices:
            repair_callback(scene, idx, 1)

    for acc_idx, anchor_idx in role_graph.accessory_to_anchor.items():
        target_pos, target_yaw = target_pose_for_attachment(scene, objs, acc_idx, anchor_idx, yaw_from_quaternion)
        cur = objs[acc_idx].get("pos", [0.0, 0.0, 0.0])
        objs[acc_idx]["pos"] = [
            (1.0 - blend) * cur[0] + blend * target_pos[0],
            cur[1],
            (1.0 - blend) * cur[2] + blend * target_pos[2],
        ]
        if role_graph.categories[acc_idx] in {"chair", "nightstand", "lamp", "mirror"}:
            objs[acc_idx]["rot"] = quaternion_from_yaw(target_yaw)
        repair_callback(scene, acc_idx, 1)

    if full_repair_passes > 0:
        for i in range(len(objs)):
            repair_callback(scene, i, full_repair_passes)
