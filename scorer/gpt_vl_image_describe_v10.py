from __future__ import annotations

import base64
import json
import math
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter


@dataclass
class GPTMovePromptV5Result:
    raw_text: str
    move_prompt: str


@dataclass
class GPTRelationPriorsV5Result:
    raw_text: str
    json_text: str


@dataclass
class ObjectEdit:
    jid_prefix: str
    description: str
    hint_pos: Optional[List[float]] = None
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    target_yaw_deg: Optional[float] = None
    relative_yaw_deg: Optional[float] = None
    no_movement: bool = False
    no_rotation: bool = False
    raw_line: str = ""


@dataclass
class MovePromptParseResult:
    room_name: str = ""
    edits: List[ObjectEdit] = field(default_factory=list)
    header_line: str = ""
    parse_warnings: List[str] = field(default_factory=list)


def _scene_signature(scene: Dict[str, Any]) -> str:
    return json.dumps(scene, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _compact_scene_for_prompt(scene: Dict[str, Any]) -> Dict[str, Any]:
    objects: List[Dict[str, Any]] = []
    for i, obj in enumerate(scene.get("objects", [])):
        objects.append(
            {
                "idx": i,
                "jid": obj.get("sampled_asset_jid") or obj.get("jid") or obj.get("sampled_jid"),
                "category": obj.get("category") or obj.get("type"),
                "desc": obj.get("description") or obj.get("style_description") or obj.get("desc"),
                "pos": obj.get("pos"),
                "rot": obj.get("rot"),
                "size": obj.get("size"),
            }
        )
    return {
        "room_type": scene.get("room_type"),
        "bounds_bottom": scene.get("bounds_bottom"),
        "bounds_top": scene.get("bounds_top"),
        "objects": objects,
    }


@lru_cache(maxsize=64)
def _img_to_data_url_cached(path_str: str, mtime_ns: int, size: int) -> str:
    p = Path(path_str)
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    suffix = p.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _img_to_data_url(p: Path) -> str:
    stat = p.stat()
    return _img_to_data_url_cached(str(p.resolve()), stat.st_mtime_ns, stat.st_size)


def _post_chat_completions(
    api_base: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout_s: float,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    http = session or requests
    resp = http.post(url, headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def _strip_markdown_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _extract_json_text(text: str) -> str:
    cleaned = _strip_markdown_fence(text)
    try:
        json.loads(cleaned)
        return cleaned
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for i, ch in enumerate(cleaned):
        if ch not in '{[':
            continue
        try:
            _, end = decoder.raw_decode(cleaned[i:])
            return cleaned[i : i + end]
        except Exception:
            continue
    raise ValueError('Model output does not contain valid JSON.')


def _normalize_relation_priors_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, list):
        return {"relations": payload}
    if isinstance(payload, dict):
        if isinstance(payload.get("relations"), list):
            return payload
        for key in ("relation_priors", "priors", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return {"relations": value}
        return {"relations": []}
    return {"relations": []}


def yaw_from_quaternion(q: List[float]) -> float:
    x, y, z, w = q
    siny_cosp = 2.0 * (w * y + z * x)
    cosy_cosp = 1.0 - 2.0 * (y * y + x * x)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def quaternion_from_yaw(yaw_deg: float) -> List[float]:
    yaw_rad = math.radians(yaw_deg)
    return [0.0, math.sin(yaw_rad / 2.0), 0.0, math.cos(yaw_rad / 2.0)]


def _make_object_label(obj: Dict[str, Any], index: int, counts: Dict[str, int], seen: Dict[str, int]) -> str:
    jid = obj.get("sampled_asset_jid") or obj.get("jid") or obj.get("sampled_jid") or "unknown"
    prefix = jid[:6].lower() if len(jid) >= 6 else jid.lower()

    category = obj.get("category") or obj.get("type") or ""
    if not category:
        desc = obj.get("description") or obj.get("style_description") or obj.get("desc") or ""
        words = desc.split()[:3]
        category = " ".join(words) if words else f"Object_{index}"

    pos = obj.get("pos", [0.0, 0.0, 0.0])
    x = float(pos[0]) if len(pos) > 0 else 0.0
    z = float(pos[2]) if len(pos) > 2 else 0.0

    if counts.get(prefix, 1) > 1:
        rank = seen.get(prefix, 0) + 1
        seen[prefix] = rank
        return f"{category} #{rank} ({prefix}…) at ({x:.2f}, {z:.2f})"
    return f"{category} ({prefix}…) at ({x:.2f}, {z:.2f})"


def build_labeled_scene_summary(scene: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    objs = scene.get("objects", [])
    counts: Dict[str, int] = {}
    for obj in objs:
        jid = obj.get("sampled_asset_jid") or obj.get("jid") or obj.get("sampled_jid") or ""
        prefix = jid[:6].lower() if len(jid) >= 6 else jid.lower()
        counts[prefix] = counts.get(prefix, 0) + 1

    seen: Dict[str, int] = {}
    lines: List[str] = []
    labeled: List[Dict[str, Any]] = []
    for i, obj in enumerate(objs):
        label = _make_object_label(obj, i, counts, seen)
        pos = obj.get("pos", [0.0, 0.0, 0.0])
        rot = obj.get("rot", [0.0, 0.0, 0.0, 1.0])
        yaw = yaw_from_quaternion(rot) if isinstance(rot, list) and len(rot) == 4 else 0.0
        lines.append(
            f"  - {label} | pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | yaw={yaw:.1f}°"
        )
        labeled.append({**obj, "_label": label})
    return "OBJECTS IN SCENE:\n" + "\n".join(lines), labeled


_RE_JID = re.compile(r"\(([a-f0-9]{6})[^)]*\)", re.IGNORECASE)
_RE_AT_POS = re.compile(
    r"at\s*\(\s*([+\-−]?\d+(?:\.\d+)?)\s*,\s*([+\-−]?\d+(?:\.\d+)?)\s*(?:,\s*([+\-−]?\d+(?:\.\d+)?))?\s*\)",
    re.IGNORECASE,
)
_RE_MOVE_SEGMENT = re.compile(
    r"move\s+([+\-−]?\s*\d+(?:\.\d+)?)\s*m\s+along\s+([+\-−]?[XYZxyz])",
    re.IGNORECASE,
)
_RE_ROT_ABSOLUTE = re.compile(r"rotate\s+to\s+(?:face\s+)?([+\-−]?\d+(?:\.\d+)?)\s*°?", re.IGNORECASE)
_RE_ROT_RELATIVE = re.compile(
    r"rotate\s+([+\-−]?\d+(?:\.\d+)?)\s*°?\s*(clockwise|counter[- ]?clockwise|ccw|cw)?",
    re.IGNORECASE,
)
_RE_NO_MOVEMENT = re.compile(r"no\s+movement\s+needed", re.IGNORECASE)
_RE_NO_ROTATION = re.compile(r"no\s+rotation\s+needed", re.IGNORECASE)


def _parse_sign_number(text: str) -> float:
    return float(text.strip().replace("−", "-").replace(" ", ""))


def _parse_axis_sign(text: str) -> Tuple[str, float]:
    text = text.strip().replace("−", "-")
    if text.startswith("+"):
        return text[1:].upper(), 1.0
    if text.startswith("-"):
        return text[1:].upper(), -1.0
    return text.upper(), 1.0


def parse_move_prompt(text: str) -> MovePromptParseResult:
    result = MovePromptParseResult()
    lines = text.strip().splitlines()

    object_lines: List[str] = []
    found_header = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not found_header:
            if "coordinate convention" in stripped.lower() or "precise pos + rot edits" in stripped.lower():
                result.header_line = stripped
                found_header = True
                continue
            if "adjust" in stripped.lower() and "edits" in stripped.lower():
                result.header_line = stripped
                found_header = True
                continue
            continue

        if not result.room_name and not _RE_JID.search(stripped):
            result.room_name = stripped
            continue

        if _RE_JID.search(stripped):
            object_lines.append(stripped)
        elif object_lines:
            object_lines[-1] += " " + stripped
        else:
            result.parse_warnings.append(f"unrecognized line before any object: {stripped}")

    for line in object_lines:
        jid_match = _RE_JID.search(line)
        if not jid_match:
            result.parse_warnings.append(f"no jid found: {line}")
            continue

        edit = ObjectEdit(
            jid_prefix=jid_match.group(1).lower(),
            description=line[: jid_match.start()].strip().rstrip("(").strip(),
            raw_line=line,
        )

        at_match = _RE_AT_POS.search(line)
        if at_match:
            x = _parse_sign_number(at_match.group(1))
            y_or_z = _parse_sign_number(at_match.group(2))
            if at_match.group(3) is not None:
                z = _parse_sign_number(at_match.group(3))
                edit.hint_pos = [x, y_or_z, z]
            else:
                edit.hint_pos = [x, 0.0, y_or_z]

        if _RE_NO_MOVEMENT.search(line):
            edit.no_movement = True
        else:
            for match in _RE_MOVE_SEGMENT.finditer(line):
                amount = _parse_sign_number(match.group(1))
                axis, sign = _parse_axis_sign(match.group(2))
                delta = amount * sign
                if axis == "X":
                    edit.dx += delta
                elif axis == "Y":
                    edit.dy += delta
                elif axis == "Z":
                    edit.dz += delta

        if _RE_NO_ROTATION.search(line):
            edit.no_rotation = True
        else:
            abs_match = _RE_ROT_ABSOLUTE.search(line)
            if abs_match:
                edit.target_yaw_deg = _parse_sign_number(abs_match.group(1))
            else:
                rel_match = _RE_ROT_RELATIVE.search(line)
                if rel_match:
                    deg = _parse_sign_number(rel_match.group(1))
                    direction = (rel_match.group(2) or "").lower().replace(" ", "").replace("-", "")
                    if "counter" in direction or "ccw" in direction:
                        deg = -abs(deg)
                    elif "clockwise" in direction or "cw" in direction:
                        deg = abs(deg)
                    edit.relative_yaw_deg = deg

        result.edits.append(edit)

    return result


def _pos_distance_xz(a: List[float], b: List[float]) -> float:
    return math.hypot(a[0] - b[0], (a[2] if len(a) > 2 else 0.0) - (b[2] if len(b) > 2 else 0.0))


def apply_edits_to_scene(scene: Dict[str, Any], edits: List[ObjectEdit]) -> Tuple[Dict[str, Any], int, List[Dict[str, Any]]]:
    edits_by_prefix: Dict[str, List[ObjectEdit]] = {}
    for edit in edits:
        edits_by_prefix.setdefault(edit.jid_prefix, []).append(edit)

    objects = scene.get("objects", [])
    objs_by_prefix: Dict[str, List[int]] = {}
    for i, obj in enumerate(objects):
        jid = obj.get("sampled_asset_jid") or obj.get("jid") or obj.get("sampled_jid")
        if isinstance(jid, str) and len(jid) >= 6:
            objs_by_prefix.setdefault(jid[:6].lower(), []).append(i)

    applied = 0
    changes: List[Dict[str, Any]] = []

    for prefix, edit_list in edits_by_prefix.items():
        obj_indices = list(objs_by_prefix.get(prefix, []))
        if not obj_indices:
            continue

        pairs: List[Tuple[ObjectEdit, int]] = []
        available = set(obj_indices)

        with_pos = [e for e in edit_list if e.hint_pos is not None]
        without_pos = [e for e in edit_list if e.hint_pos is None]

        for edit in with_pos:
            if not available:
                break
            best_idx = min(available, key=lambda oi: _pos_distance_xz(edit.hint_pos or [0.0, 0.0, 0.0], objects[oi].get("pos", [0.0, 0.0, 0.0])))
            pairs.append((edit, best_idx))
            available.discard(best_idx)

        for edit, oi in zip(without_pos, sorted(available)):
            pairs.append((edit, oi))

        for edit, obj_idx in pairs:
            obj = objects[obj_idx]
            before_pos = list(obj.get("pos", [0.0, 0.0, 0.0]))
            before_rot = list(obj.get("rot", [0.0, 0.0, 0.0, 1.0]))
            changed = False

            if not edit.no_movement and (abs(edit.dx) > 1e-9 or abs(edit.dy) > 1e-9 or abs(edit.dz) > 1e-9):
                obj["pos"] = [before_pos[0] + edit.dx, before_pos[1] + edit.dy, before_pos[2] + edit.dz]
                changes.append(
                    {
                        "obj_index": obj_idx,
                        "field": "pos",
                        "delta": [edit.dx, edit.dy, edit.dz],
                        "before": before_pos,
                        "after": obj["pos"],
                    }
                )
                changed = True

            if not edit.no_rotation:
                target_yaw = None
                if edit.target_yaw_deg is not None:
                    target_yaw = edit.target_yaw_deg
                elif edit.relative_yaw_deg is not None:
                    target_yaw = yaw_from_quaternion(before_rot) + edit.relative_yaw_deg
                if target_yaw is not None:
                    obj["rot"] = quaternion_from_yaw(target_yaw)
                    changes.append(
                        {
                            "obj_index": obj_idx,
                            "field": "rot",
                            "before": before_rot,
                            "after": obj["rot"],
                            "target_yaw_deg": target_yaw,
                        }
                    )
                    changed = True

            if changed:
                applied += 1

    return scene, applied, changes


class GPTVLMovePromptGeneratorV5:
    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        self.api_base = (api_base or os.getenv("YUNWU_AI_API_BASE") or "").rstrip("/")
        self.api_key = api_key or os.getenv("YUNWU_AI_API_KEY")
        self.timeout_s = timeout_s
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=4, max_retries=0)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self._prompt_cache: Dict[str, str] = {}
        self._relation_prompt_cache: Dict[str, str] = {}
        if not self.api_key:
            raise RuntimeError("Missing YUNWU_AI_API_KEY.")
        if not self.api_base:
            raise RuntimeError("Missing YUNWU_AI_API_BASE.")

    @staticmethod
    def _build_prompt(scene: Dict[str, Any], extra_context: str) -> str:
        object_summary, _ = build_labeled_scene_summary(scene)
        scene_str = json.dumps(_compact_scene_for_prompt(scene), ensure_ascii=False, separators=(",", ":"))
        return f"""You are a professional 3D indoor scene layout analyst.

You will be given:
- Image 1: diagonal view render
- Image 2: top-down annotated render
- A compact scene JSON with exact pos/rot values
- A labeled object list

Goal:
Output precise move and rotation edits that improve the layout.
Priorities:
1. Fix out-of-bounds
2. Fix collisions
3. Align wall-side furniture with walls
4. Make chairs face their functional anchors
5. Keep small bedside tables near the bed

Output format (must follow exactly):
Line 1:
Use the top-view coordinate convention +X = right, −X = left, +Z = forward, −Z = backward. Adjust the current scene toward the reference layout with the following precise pos + rot edits:

Line 2: blank
Line 3: room/area name
Then one blank line.
Then exactly one line per object:
<short_label> (<jid6>…) at (<x>, <z>): <movement instruction>; <rotation instruction>.

Movement instruction must be one of:
- no movement needed
- move +0.20 m along +X
- move +0.20 m along +X and −0.15 m along +Z

Rotation instruction must be one of:
- no rotation needed
- rotate to face 270°
- rotate 30° clockwise
- rotate 15° counter-clockwise

Rules:
- Use one line per object.
- Use the current x,z from JSON in the at(...) field.
- Round movement to 0.05 m.
- Keep the output concise.

OBJECTS:
{object_summary}

ADDITIONAL CONTEXT:
{extra_context.strip() if extra_context else '(none)'}

SCENE_JSON:
{scene_str}
"""

    @staticmethod
    def _build_relation_priors_prompt(scene: Dict[str, Any], extra_context: str) -> str:
        object_summary, _ = build_labeled_scene_summary(scene)
        scene_str = json.dumps(_compact_scene_for_prompt(scene), ensure_ascii=False, separators=(",", ":"))
        return f"""You are a professional 3D indoor scene layout analyst.

You will be given:
- Image 1: diagonal view render
- Image 2: top-down annotated render
- A compact scene JSON with exact pos/rot values
- A labeled object list

Goal:
Infer a small set of high-confidence relation priors that describe the intended functional layout.

Allowed relation types only:
- near
- distance_band
- facing
- facing_pair
- centered_with
- in_front_of
- side_of
- against_wall
- parallel

Output must be strict JSON, with no markdown fence and no commentary:
{{
  "relations": [
    {{"src_idx": 0, "tgt_idx": 1, "type": "near", "confidence": 0.82, "weight": 1.0, "reason": "chair near desk"}},
    {{"src_idx": 2, "type": "against_wall", "confidence": 0.91, "weight": 1.1, "reason": "wardrobe against wall"}}
  ]
}}

Rules:
- Use object indices from SCENE_JSON.
- Use at most 3 relations per source object.
- Prefer high-confidence, functionally meaningful relations only.
- Do not invent missing objects.
- For against_wall and parallel, omit tgt_idx.
- confidence should be in [0.0, 1.0].
- weight should usually be in [0.5, 1.5].
- If uncertain, return fewer relations, even an empty list.

Heuristics:
- Beds, wardrobes, cabinets, shelves, TV stands, and narrow consoles often relate to walls.
- Chairs often relate to desks, tables, or sofas.
- Coffee tables often sit in front of sofas.
- Nightstands often stay near and to the side of beds.
- Tables often align parallel to nearby walls.

OBJECTS:
{object_summary}

ADDITIONAL CONTEXT:
{extra_context.strip() if extra_context else '(none)'}

SCENE_JSON:
{scene_str}
"""

    def _chat(
        self,
        *,
        diag_image_path: Path,
        top_image_path: Path,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        if not diag_image_path.exists():
            raise FileNotFoundError(diag_image_path)
        if not top_image_path.exists():
            raise FileNotFoundError(top_image_path)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _img_to_data_url(diag_image_path)}},
                        {"type": "image_url", "image_url": {"url": _img_to_data_url(top_image_path)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        data = _post_chat_completions(
            self.api_base,
            self.api_key,
            payload,
            self.timeout_s,
            session=self.session,
        )
        return data["choices"][0]["message"]["content"]

    def generate(
        self,
        *,
        diag_image_path: Path,
        top_image_path: Path,
        scene: Dict[str, Any],
        extra_context: str = "",
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> GPTMovePromptV5Result:
        cache_key = _scene_signature(scene) + "\n<context>\n" + extra_context
        prompt = self._prompt_cache.get(cache_key)
        if prompt is None:
            prompt = self._build_prompt(scene, extra_context)
            self._prompt_cache[cache_key] = prompt
        raw_text = self._chat(
            diag_image_path=diag_image_path,
            top_image_path=top_image_path,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return GPTMovePromptV5Result(raw_text=raw_text, move_prompt=_strip_markdown_fence(raw_text))

    def generate_relation_priors(
        self,
        *,
        diag_image_path: Path,
        top_image_path: Path,
        scene: Dict[str, Any],
        extra_context: str = "",
        temperature: float = 0.0,
        max_tokens: int = 900,
    ) -> GPTRelationPriorsV5Result:
        cache_key = _scene_signature(scene) + "\n<relation_context>\n" + extra_context
        prompt = self._relation_prompt_cache.get(cache_key)
        if prompt is None:
            prompt = self._build_relation_priors_prompt(scene, extra_context)
            self._relation_prompt_cache[cache_key] = prompt

        raw_text = self._chat(
            diag_image_path=diag_image_path,
            top_image_path=top_image_path,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        json_text = _extract_json_text(raw_text)
        payload = _normalize_relation_priors_payload(json.loads(json_text))
        normalized_json_text = json.dumps(payload, ensure_ascii=False)
        return GPTRelationPriorsV5Result(raw_text=raw_text, json_text=normalized_json_text)
