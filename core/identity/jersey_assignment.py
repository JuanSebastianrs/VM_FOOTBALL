"""
Per-team jersey number assignment with roster mask and temporal duplicate resolution.

Replaces the original Hungarian global approach (which was destructive due to
ByteTrack fragmentation). New strategy:

1. Roster Mask: if roster provided, zero out probabilities for numbers not in roster.
2. Top-1 selection: each tracklet gets its highest-probability number.
3. Temporal Duplicate Resolution: if two tracklets have the same number AND
   overlapping frames, keep the higher-confidence one; disjoint = allow both.
4. Lock/Confidence: calibrated thresholds for locked/tentative/unknown states.

Usage:
    from core.identity.jersey_assignment import assign_per_team
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


class TrackletInfo:
    """Container for tracklet data used in jersey assignment."""

    def __init__(self, track_id, team_id, jersey_probs, num_frames=0,
                 quality_scores=None, state="unknown", jersey_number=None,
                 confidence=0.0, is_goalkeeper=False, frame_ids=None):
        self.track_id = track_id
        self.team_id = team_id
        self.jersey_probs = jersey_probs  # shape (99,) for numbers 1-99
        self.num_frames = num_frames
        self.quality_scores = quality_scores or []
        self.state = state
        self.jersey_number = jersey_number
        self.confidence = confidence
        self.is_goalkeeper = is_goalkeeper
        self.frame_ids = frame_ids or []


def apply_roster_mask(jersey_probs, roster_set):
    """
    Zero out probabilities for numbers not in the roster, then re-normalize.

    Args:
        jersey_probs: np.array of shape (99,), probs for numbers 1-99.
        roster_set: set of allowed jersey numbers (e.g., {1, 4, 7, 10, ...}).

    Returns:
        Masked and re-normalized probability array.
    """
    if jersey_probs is None or len(jersey_probs) < 99:
        return jersey_probs

    masked = jersey_probs.copy()
    for idx in range(99):
        num = idx + 1
        if num not in roster_set:
            masked[idx] = 0.0

    total = masked.sum()
    if total > 1e-8:
        masked = masked / total
    else:
        # All roster numbers had zero probability — leave as zeros
        masked = np.zeros(99, dtype=np.float64)

    return masked


def select_top1(jersey_probs):
    """
    Select the top-1 jersey number from probability distribution.

    Returns:
        (number, confidence) or (None, 0.0) if no valid prediction.
    """
    if jersey_probs is None or len(jersey_probs) < 99:
        return None, 0.0

    best_idx = int(np.argmax(jersey_probs))
    best_prob = float(jersey_probs[best_idx])

    if best_prob < 1e-6:
        return None, 0.0

    return best_idx + 1, best_prob


def compute_lock_state(jersey_probs, assigned_number,
                       p1_threshold=0.60, margin_threshold=0.20):
    """
    Determine lock state based on confidence and margin over second-best.

    Returns:
        (state, confidence, margin)
    """
    if jersey_probs is None or len(jersey_probs) < 99 or assigned_number is None:
        return "unknown", 0.0, 0.0

    p_assigned = float(jersey_probs[assigned_number - 1])

    # Margin against the next-best number
    sorted_probs = sorted(jersey_probs, reverse=True)
    p2_val = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin = p_assigned - p2_val

    if p_assigned >= p1_threshold and margin >= margin_threshold:
        state = "locked"
    elif p_assigned >= 0.40 and margin >= 0.10:
        state = "tentative"
    else:
        state = "unknown"

    return state, p_assigned, margin


def frames_overlap(frames_a, frames_b):
    """Check if two frame lists have any overlap."""
    if not frames_a or not frames_b:
        return False
    return bool(set(frames_a) & set(frames_b))


def resolve_temporal_duplicates(assignments, tracklets, reassign_conflicts=False,
                                min_alternative_prob=0.30):
    """
    Resolve duplicate jersey numbers within a team using temporal overlap.

    Rules:
    - Two tracklets with same number and OVERLAPPING frames: keep higher confidence.
    - Two tracklets with same number and DISJOINT frames: allow both (fragmentation).
    - This replaces the destructive Hungarian global unicidad constraint.
    - reassign_conflicts=True: the losing tracklet falls back to its best
      alternative number that does not conflict with any overlapping kept
      tracklet (intra-frame mutual exclusivity), committed as 'tentative' if
      its probability >= min_alternative_prob; otherwise it goes 'unknown'.

    Args:
        assignments: dict {track_id: assignment_dict}
        tracklets: list of TrackletInfo objects

    Returns:
        Updated assignments dict.
    """
    # Build lookup: number -> list of (track_id, confidence, frame_ids)
    tracklet_lookup = {t.track_id: t for t in tracklets}
    number_to_tracks = {}

    for tid, asn in assignments.items():
        num = asn.get("predicted_number")
        if num is None:
            continue
        t = tracklet_lookup.get(tid)
        frame_ids = t.frame_ids if t else []
        conf = asn.get("confidence", 0.0)

        if num not in number_to_tracks:
            number_to_tracks[num] = []
        number_to_tracks[num].append((tid, conf, frame_ids))

    resolved = dict(assignments)  # copy

    for num, track_list in number_to_tracks.items():
        if len(track_list) <= 1:
            continue

        # Sort by confidence descending
        track_list.sort(key=lambda x: x[1], reverse=True)

        # Check pairwise overlaps — winner keeps the number, losers with overlap go unknown
        kept = set()
        for i, (tid_i, conf_i, frames_i) in enumerate(track_list):
            conflict_tid = None
            for tid_k in kept:
                t_k = tracklet_lookup.get(tid_k)
                if t_k and frames_overlap(frames_i, t_k.frame_ids):
                    conflict_tid = tid_k
                    break

            if i == 0:
                # Highest confidence always keeps
                kept.add(tid_i)
            elif conflict_tid is not None:
                # Overlapping with a higher-confidence holder
                fallback = None
                if reassign_conflicts:
                    fallback = _best_non_conflicting_alternative(
                        tracklet_lookup.get(tid_i), frames_i, resolved,
                        tracklet_lookup, min_alternative_prob,
                    )
                if fallback is not None:
                    alt_num, alt_prob = fallback
                    resolved[tid_i] = {
                        **resolved[tid_i],
                        "predicted_number": alt_num,
                        "confidence": alt_prob,
                        "state": "tentative",
                        "duplicate_conflict_with": conflict_tid,
                        "reassigned_from": num,
                    }
                else:
                    resolved[tid_i] = {
                        **resolved[tid_i],
                        "predicted_number": None,
                        "state": "unknown",
                        "duplicate_conflict_with": conflict_tid,
                    }
            else:
                # Disjoint frames — allow (likely same player, fragmented track)
                kept.add(tid_i)

    return resolved


def _best_non_conflicting_alternative(tracklet, frames, resolved, tracklet_lookup,
                                      min_prob):
    """
    Best alternative number for a conflict loser: highest-probability number
    not already used by any temporally-overlapping tracklet of the same team.
    """
    if tracklet is None or tracklet.jersey_probs is None:
        return None
    probs = tracklet.jersey_probs
    # Numbers already committed by overlapping tracklets
    taken = set()
    for other_tid, other_asn in resolved.items():
        if other_tid == tracklet.track_id:
            continue
        other_num = other_asn.get("predicted_number")
        if other_num is None:
            continue
        other_t = tracklet_lookup.get(other_tid)
        if other_t is not None and frames_overlap(frames, other_t.frame_ids):
            taken.add(other_num)

    order = np.argsort(probs)[::-1]
    for idx in order[1:]:  # skip top-1 (the conflicting number)
        num = int(idx) + 1
        p = float(probs[idx])
        if p < min_prob:
            return None
        if num not in taken:
            return num, p
    return None


def assign_jersey_numbers_simple(
    tracklets: List[TrackletInfo],
    team_id: int,
    roster: Optional[set] = None,
    p1_threshold: float = 0.60,
    margin_threshold: float = 0.20,
) -> Dict[int, Dict]:
    """
    Assign jersey numbers to tracklets using roster mask + top-1 selection.
    NO Hungarian — each tracklet independently gets its best number.

    Args:
        tracklets: List of TrackletInfo objects for this team.
        team_id: Team identifier (0 or 1).
        roster: Optional set of allowed jersey numbers.
        p1_threshold: Min probability for locked state.
        margin_threshold: Min margin between top-2 for locked.

    Returns:
        Dict mapping track_id -> assignment dict.
    """
    results = {}

    for t in tracklets:
        probs = t.jersey_probs

        # Apply roster mask if provided
        if roster and probs is not None:
            probs = apply_roster_mask(probs, roster)

        # Select top-1
        number, confidence = select_top1(probs)

        # Determine lock state
        state, p1, margin = compute_lock_state(
            probs, number,
            p1_threshold=p1_threshold,
            margin_threshold=margin_threshold,
        )

        # Only commit number if state is locked or tentative
        if state == "unknown":
            committed_number = None
        else:
            committed_number = number

        results[t.track_id] = {
            "track_id": t.track_id,
            "team_id": team_id,
            "predicted_number": committed_number,
            "confidence": confidence,
            "state": state,
            "p1": p1,
            "margin": margin,
            "is_goalkeeper": t.is_goalkeeper,
            "raw_top1": number,  # always include raw prediction for evaluation
        }

    return results


def assign_per_team(
    all_tracklets: List[TrackletInfo],
    team_rosters: Optional[Dict[int, set]] = None,
    duplicate_penalty: float = 1.0,  # kept for API compat, unused
    p1_threshold: float = 0.60,
    margin_threshold: float = 0.20,
    reassign_conflicts: bool = False,
) -> Dict[int, Dict]:
    """
    Top-level entry point: partition tracklets by team, assign numbers per-team,
    resolve temporal duplicates, merge results.

    Args:
        all_tracklets: All tracklets from a sequence.
        team_rosters: Optional dict {team_id: set_of_numbers}.
        p1_threshold: Min probability for locked state.
        margin_threshold: Min margin between top-2 for locked.

    Returns:
        Dict mapping track_id -> final assignment dict.
    """
    team_rosters = team_rosters or {}
    by_team = {0: [], 1: []}
    excluded = []
    for t in all_tracklets:
        if t.team_id in (0, 1):
            by_team[t.team_id].append(t)
        else:
            # team_id=-1 or other unknown values: do NOT send to team 0
            excluded.append(t)

    all_assignments = {}

    # Mark excluded tracklets (unknown team) as unknown
    for t in excluded:
        all_assignments[t.track_id] = {
            "track_id": t.track_id,
            "team_id": t.team_id,
            "predicted_number": None,
            "confidence": 0.0,
            "state": "unknown",
            "is_goalkeeper": t.is_goalkeeper,
        }

    for team_id, team_tracklets in by_team.items():
        roster = team_rosters.get(team_id)
        if not team_tracklets:
            continue

        # Step 1: Roster mask + top-1 assignment
        raw_assignments = assign_jersey_numbers_simple(
            team_tracklets, team_id, roster=roster,
            p1_threshold=p1_threshold, margin_threshold=margin_threshold,
        )

        # Step 2: Resolve temporal duplicates
        resolved = resolve_temporal_duplicates(raw_assignments, team_tracklets,
                                               reassign_conflicts=reassign_conflicts)

        all_assignments.update(resolved)

    return all_assignments