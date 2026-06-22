"""
Tracklet fragment linking for jersey identity.

ByteTrack fragments a single player into several track_ids. Each fragment in
isolation has less temporal evidence, which costs jersey accuracy (E2E v1.5 on
SNMOT-148: best-fragment raw 77.8% vs representative-fragment 72.2%).

This module links fragments that very likely belong to the same player using
ONLY pipeline-internal signals (no GT):
  - same team_id,
  - temporally disjoint with a bounded gap (a player cannot be two tracklets
    at the same time; huge gaps are too risky to bridge),
  - spatial continuity: the exit bbox of the earlier fragment and the entry
    bbox of the later fragment are close in image space, with a tolerance that
    grows with the temporal gap.

Linking is greedy on (gap, distance): shortest, closest joins first; each
fragment joins at most one chain on each side.
"""

from collections import defaultdict


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _bbox_size(bbox):
    x1, y1, x2, y2 = bbox
    return max(x2 - x1, y2 - y1)


def link_fragments(tracklets, max_gap=50, dist_per_frame=6.0, base_dist=40.0,
                   max_dist=300.0, fragment_summary=None, uncertain_p1=0.0):
    """
    Group tracklet fragments into identity chains.

    Args:
        tracklets: list of dicts with keys
            track_id, team_id, all_frame_ids (sorted asc), all_bboxes.
        max_gap: maximum frame gap to bridge between fragments.
        dist_per_frame: extra pixel tolerance per frame of gap
            (players move; ~6 px/frame covers a sprint at broadcast scale).
        base_dist: base pixel tolerance at gap=1.
        max_dist: hard cap on the spatial tolerance (large gaps must not
            bridge across half the screen).
        fragment_summary: optional dict track_id -> (top1_number, p1_conf)
            from per-fragment fusion. When given, a link additionally requires
            identity compatibility: both fragments must predict the SAME
            number (or one has no evidence at all). Pooling fragments that
            disagree is never allowed — an uncertain wrong fragment can flip a
            confident correct one (observed on SNMOT-132 GT#33).
        uncertain_p1: unused when strict agreement is enforced; kept for
            experimentation (set > 0 to also allow links where either
            fragment's confidence is below this value).

    Returns:
        dict track_id -> group_id, where fragments sharing a group_id are
        considered the same player identity.
    """
    frags = []
    for t in tracklets:
        fids = t.get("all_frame_ids") or t.get("frame_ids") or []
        boxes = t.get("all_bboxes") or t.get("bboxes") or []
        if not fids or not boxes:
            continue
        order = sorted(range(len(fids)), key=lambda i: fids[i])
        frags.append({
            "track_id": t["track_id"],
            "team_id": t.get("team_id", -1),
            "start": fids[order[0]],
            "end": fids[order[-1]],
            "start_box": boxes[order[0]],
            "end_box": boxes[order[-1]],
        })

    # Candidate links: earlier fragment end -> later fragment start
    candidates = []
    for a in frags:
        for b in frags:
            if a["track_id"] == b["track_id"]:
                continue
            if a["team_id"] != b["team_id"] or a["team_id"] not in (0, 1):
                continue
            gap = b["start"] - a["end"]
            if gap <= 0 or gap > max_gap:
                continue
            ax, ay = _bbox_center(a["end_box"])
            bx, by = _bbox_center(b["start_box"])
            dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            # Tolerance grows with gap and with player size on screen, capped
            tol = min(max_dist,
                      base_dist + dist_per_frame * gap + 0.5 * _bbox_size(a["end_box"]))
            if dist > tol:
                continue
            if fragment_summary is not None:
                sa = fragment_summary.get(a["track_id"])
                sb = fragment_summary.get(b["track_id"])
                if sa is not None and sb is not None:
                    no_evidence = sa[0] is None or sb[0] is None
                    same_number = sa[0] is not None and sa[0] == sb[0]
                    either_uncertain = (uncertain_p1 > 0.0 and
                                        (sa[1] < uncertain_p1 or sb[1] < uncertain_p1))
                    if not (same_number or no_evidence or either_uncertain):
                        continue
            candidates.append((gap, dist, a["track_id"], b["track_id"]))

    # Greedy: shortest gap, then closest distance
    candidates.sort()
    next_of = {}   # tid -> tid linked after
    prev_of = {}   # tid -> tid linked before
    for gap, dist, a_tid, b_tid in candidates:
        if a_tid in next_of or b_tid in prev_of:
            continue
        next_of[a_tid] = b_tid
        prev_of[b_tid] = a_tid

    # Collapse chains into group ids (rooted at the earliest fragment)
    group_of = {}
    for f in frags:
        tid = f["track_id"]
        if tid in group_of:
            continue
        root = tid
        seen = {root}
        while root in prev_of and prev_of[root] not in seen:
            root = prev_of[root]
            seen.add(root)
        # walk forward from root assigning the group
        cur = root
        chain_seen = set()
        while cur is not None and cur not in chain_seen:
            group_of[cur] = root
            chain_seen.add(cur)
            cur = next_of.get(cur)

    # Fragments without any link map to themselves
    for f in frags:
        group_of.setdefault(f["track_id"], f["track_id"])
    return group_of


def detect_mode_switch(frame_results, min_segment=8, dominance=0.6):
    """
    Detect a probable ID switch inside one tracklet by scanning for a split
    point where the dominant per-frame prediction changes.

    Args:
        frame_results: list of per-frame dicts with 'frame_id' and
            'jersey_probs' (sorted by frame_id).
        min_segment: minimum frames per side to consider a split.
        dominance: each side's mode must cover at least this fraction
            of its frames.

    Returns:
        (switch_index, left_mode, right_mode) where switch_index is the index
        of the first frame of the right segment, or (None, None, None) when no
        switch is detected.
    """
    n = len(frame_results)
    if n < 2 * min_segment:
        return None, None, None

    tops = [int(fr["jersey_probs"].argmax()) + 1 for fr in frame_results]

    best = None
    for split in range(min_segment, n - min_segment + 1):
        left, right = tops[:split], tops[split:]
        lm = max(set(left), key=left.count)
        rm = max(set(right), key=right.count)
        if lm == rm:
            continue
        l_dom = left.count(lm) / len(left)
        r_dom = right.count(rm) / len(right)
        if l_dom >= dominance and r_dom >= dominance:
            score = l_dom * r_dom * min(len(left), len(right))
            if best is None or score > best[0]:
                best = (score, split, lm, rm)

    if best is None:
        return None, None, None
    return best[1], best[2], best[3]
