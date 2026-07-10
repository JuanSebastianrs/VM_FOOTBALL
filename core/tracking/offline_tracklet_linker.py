"""
Offline tracklet linking and gap interpolation (association post-processing).

Takes the per-frame tracked detections produced by the online tracker
(RF-DETR + ByteTrack) and re-links fragmented tracklets after the fact,
using the whole clip at once:

1.  Camera-motion stabilization: per-frame affine CMC matrices are composed
    into a global transform so every tracklet lives in a common (frame-1)
    coordinate system. Camera pans no longer look like player motion.
2.  Tracklet linking: a tracklet that ends and one that starts shortly after
    are linked if a constant-velocity prediction (in stabilized coordinates)
    from each side lands close enough to the other, their box sizes are
    consistent, and neither the team assignment nor the role contradicts the
    match. Tracklets that coexist in any frame are never linked.
3.  Gap interpolation: after linking, missing frames inside a track are
    filled by linear interpolation in stabilized coordinates and mapped back
    to image coordinates, bounded by MAX_INTERP_GAP.

Input format is the pipeline's ``<seq>_detections.json`` (list of frames with
``players`` / ``referees`` / ``goalkeepers`` entries carrying ``track_id`` and
pixel bounds). Output is a list of MOT rows ``(frame, tid, x, y, w, h)``.
"""

from dataclasses import dataclass, field

import numpy as np

ROLE_KEYS = ("players", "referees", "goalkeepers")

# Linking
MAX_LINK_GAP = 150         # frames (6 s at 25 fps)
TAIL_FIT_POINTS = 10       # points used to estimate endpoint velocity
VEL_HORIZON = 25           # frames of constant-velocity extrapolation trusted
BASE_DIST_THRESH = 0.6     # accept distance, in units of mean box diagonal
DIST_PER_GAP_FRAME = 0.035  # extra slack per frame of gap
MAX_DIST_THRESH = 1.5      # cap on the distance threshold
MAX_HEIGHT_RATIO = 1.6     # max box-height ratio between linked endpoints

# Interpolation: only short gaps — long occlusions interpolated linearly
# produce more false positives than the false negatives they recover.
MAX_INTERP_GAP = 10        # frames

# Identity signals (optional; used when embeddings / jersey numbers are attached).
# Appearance is disabled by default: calibration on co-occurrence statistics
# (different-person pairs) vs strict motion links (same-person pairs) showed
# generic ReID embeddings do NOT separate teammates (same-team different-person
# median distance 0.117 vs same-person 0.133), and a veto at 0.45 removed more
# correct links than errors (HOTA 60.83 vs 60.90). Kit color dominates the
# embedding; the mechanism is kept for domain-fine-tuned models.
APP_WEIGHT = 0.0           # weight of appearance distance in the link cost
APP_MAX = 2.0              # veto links whose appearance distance exceeds this
APP_STRONG = 0.0           # 0 disables appearance-only long-range links
LONG_RANGE_GAP = 750       # jersey-matched links allowed up to this many frames
LONG_RANGE_BASE_COST = 2.0  # rank long-range links after motion-based ones


@dataclass
class Tracklet:
    tid: int
    frames: list = field(default_factory=list)   # sorted frame ids
    boxes: dict = field(default_factory=dict)    # frame -> (x, y, w, h) tlwh
    team_id: int = -1
    role: str = ""
    emb: object = None      # L2-normalized appearance embedding, or None
    jersey: object = None   # locked jersey number (int), or None

    @property
    def start(self):
        return self.frames[0]

    @property
    def end(self):
        return self.frames[-1]


def load_tracklets(pred_frames, team_assignments=None):
    """Group per-frame detections JSON into Tracklet objects."""
    tracklets = {}
    for fdata in pred_frames:
        frame = fdata["frame_id"]
        for role in ROLE_KEYS:
            for obj in fdata.get(role, []):
                tid = obj["track_id"]
                box = (
                    obj["x_min"],
                    obj["y_min"],
                    obj["x_max"] - obj["x_min"],
                    obj["y_max"] - obj["y_min"],
                )
                tr = tracklets.setdefault(tid, Tracklet(tid))
                if frame not in tr.boxes:
                    tr.frames.append(frame)
                tr.boxes[frame] = box

    for tr in tracklets.values():
        tr.frames.sort()

    for entry in team_assignments or []:
        tr = tracklets.get(entry.get("track_id"))
        if tr is not None:
            tr.team_id = entry.get("team_id", -1)
            tr.role = entry.get("role", "") or ""

    return sorted(tracklets.values(), key=lambda t: t.start)


def attach_embeddings(tracklets, npz_path):
    """Attach OSNet embeddings from extract_tracklet_embeddings.py output."""
    data = np.load(npz_path)
    by_tid = {int(t): e for t, e in zip(data["tids"], data["emb"])}
    for tr in tracklets:
        tr.emb = by_tid.get(tr.tid)


def attach_jerseys(tracklets, jersey_data, states=("locked",)):
    """Attach confident jersey numbers from a jersey identity phase JSON."""
    by_tid = {
        e["track_id"]: e.get("predicted_number")
        for e in jersey_data.get("tracklets", [])
        if e.get("state") in states and e.get("predicted_number") is not None
    }
    for tr in tracklets:
        tr.jersey = by_tid.get(tr.tid)


def build_global_transforms(cmc, max_frame):
    """Compose per-frame CMC affines into G[f]: frame-1 coords -> frame-f coords."""
    transforms = np.tile(np.eye(3), (max_frame + 1, 1, 1))
    step = np.eye(3)
    acc = np.eye(3)
    for f in range(1, max_frame):
        mat = (cmc or {}).get(f"{f}->{f + 1}")
        if mat is not None:
            step[:2, :] = np.asarray(mat, dtype=float)
            acc = step @ acc
        transforms[f + 1] = acc
    return transforms


def _stab_center(box, g_inv):
    """Image-space box center -> stabilized (frame-1) coordinates."""
    cx = box[0] + box[2] / 2.0
    cy = box[1] + box[3] / 2.0
    p = g_inv @ np.array([cx, cy, 1.0])
    return p[:2]


def _endpoint_velocity(tr, transforms_inv, head):
    """Constant-velocity fit (px/frame, stabilized coords) at a tracklet end."""
    frames = tr.frames[:TAIL_FIT_POINTS] if head else tr.frames[-TAIL_FIT_POINTS:]
    if len(frames) < 2:
        return np.zeros(2)
    pts = np.array([_stab_center(tr.boxes[f], transforms_inv[f]) for f in frames])
    fs = np.array(frames, dtype=float)
    vx = np.polyfit(fs, pts[:, 0], 1)[0]
    vy = np.polyfit(fs, pts[:, 1], 1)[0]
    return np.array([vx, vy])


def _diag(box):
    return float(np.hypot(box[2], box[3]))


def _link_cost(a, b, transforms_inv):
    """Cost of linking tracklet a (earlier) to b (later); None if gated out."""
    gap = b.start - a.end
    if gap < 1 or gap > LONG_RANGE_GAP:
        return None
    # Role / team gates: never merge across roles or across teams.
    if a.role and b.role and a.role != b.role:
        return None
    if a.team_id >= 0 and b.team_id >= 0 and a.team_id != b.team_id:
        return None

    # Jersey gate: confident numbers that differ can never be the same player.
    both_jerseys = a.jersey is not None and b.jersey is not None
    if both_jerseys and a.jersey != b.jersey:
        return None
    same_jersey = both_jerseys and a.jersey == b.jersey

    # Appearance gate.
    app = None
    if a.emb is not None and b.emb is not None:
        app = 1.0 - float(np.dot(a.emb, b.emb))
        if app > APP_MAX:
            return None

    box_a, box_b = a.boxes[a.end], b.boxes[b.start]
    h_a, h_b = max(box_a[3], 1.0), max(box_b[3], 1.0)
    if max(h_a, h_b) / min(h_a, h_b) > MAX_HEIGHT_RATIO:
        return None

    if gap > MAX_LINK_GAP:
        # Long-range: motion is uninformative; only identity evidence links.
        if not (same_jersey or (app is not None and app <= APP_STRONG)):
            return None
        app_term = app if app is not None else APP_STRONG
        return LONG_RANGE_BASE_COST + app_term + gap * 1e-4

    p_a = _stab_center(box_a, transforms_inv[a.end])
    p_b = _stab_center(box_b, transforms_inv[b.start])
    horizon = min(gap, VEL_HORIZON)
    fwd = p_a + _endpoint_velocity(a, transforms_inv, head=False) * horizon
    bwd = p_b - _endpoint_velocity(b, transforms_inv, head=True) * horizon
    err = 0.5 * (np.linalg.norm(fwd - p_b) + np.linalg.norm(bwd - p_a))

    diag = 0.5 * (_diag(box_a) + _diag(box_b))
    err_norm = err / max(diag, 1.0)
    thresh = min(BASE_DIST_THRESH + DIST_PER_GAP_FRAME * gap, MAX_DIST_THRESH)
    if err_norm > thresh:
        return None
    return err_norm + APP_WEIGHT * (app if app is not None else 0.0)


def link_tracklets(tracklets, transforms):
    """Greedy lowest-cost linking; returns {tid -> canonical tid} chains."""
    transforms_inv = np.linalg.inv(transforms)
    candidates = []
    for a in tracklets:
        for b in tracklets:
            if a.tid == b.tid:
                continue
            cost = _link_cost(a, b, transforms_inv)
            if cost is not None:
                candidates.append((cost, a.tid, b.tid))
    candidates.sort()

    succ, pred = {}, {}
    for _, a_tid, b_tid in candidates:
        if a_tid in succ or b_tid in pred:
            continue
        succ[a_tid] = b_tid
        pred[b_tid] = a_tid

    # Collapse chains onto the id of the earliest tracklet.
    canonical = {}
    for tr in tracklets:
        if tr.tid in pred:
            continue
        tid = tr.tid
        canonical[tid] = tr.tid
        while tid in succ:
            tid = succ[tid]
            canonical[tid] = tr.tid
    return canonical


def merge_and_interpolate(tracklets, canonical, transforms):
    """Merge linked tracklets and fill bounded gaps; returns MOT rows."""
    transforms_inv = np.linalg.inv(transforms)
    merged = {}
    for tr in tracklets:
        cid = canonical[tr.tid]
        m = merged.setdefault(cid, Tracklet(cid, team_id=tr.team_id, role=tr.role))
        for f in tr.frames:
            if f not in m.boxes:
                m.frames.append(f)
            m.boxes[f] = tr.boxes[f]

    rows = []
    for m in merged.values():
        m.frames.sort()
        for f in m.frames:
            rows.append((f, m.tid) + tuple(m.boxes[f]))
        for f0, f1 in zip(m.frames, m.frames[1:]):
            gap = f1 - f0
            if gap <= 1 or gap > MAX_INTERP_GAP:
                continue
            b0, b1 = m.boxes[f0], m.boxes[f1]
            p0 = _stab_center(b0, transforms_inv[f0])
            p1 = _stab_center(b1, transforms_inv[f1])
            for f in range(f0 + 1, f1):
                t = (f - f0) / gap
                c = transforms[f] @ np.array([*(p0 + t * (p1 - p0)), 1.0])
                w = b0[2] + t * (b1[2] - b0[2])
                h = b0[3] + t * (b1[3] - b0[3])
                rows.append((f, m.tid, c[0] - w / 2.0, c[1] - h / 2.0, w, h))

    rows.sort()
    return rows


def postprocess_sequence(pred_frames, cmc=None, team_assignments=None,
                         embeddings_npz=None, jersey_data=None):
    """Full pipeline: detections JSON frames -> linked+interpolated MOT rows."""
    tracklets = load_tracklets(pred_frames, team_assignments)
    if not tracklets:
        return []
    if embeddings_npz is not None:
        attach_embeddings(tracklets, embeddings_npz)
    if jersey_data is not None:
        attach_jerseys(tracklets, jersey_data)
    max_frame = max(tr.end for tr in tracklets)
    transforms = build_global_transforms(cmc, max_frame)
    canonical = link_tracklets(tracklets, transforms)
    return merge_and_interpolate(tracklets, canonical, transforms)
