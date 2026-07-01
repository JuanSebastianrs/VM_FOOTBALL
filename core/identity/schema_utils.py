"""
Shared parsing utilities for jersey pipeline scripts.
Centralizes bbox extraction, detection/team JSON schema handling, and frame iteration
to avoid divergent parsing across core/identity/jersey_identity_phase.py,
scripts/evaluate_jersey_e2e.py, and scripts/jersey_visual_qa.py.
"""
from typing import Optional, List, Dict, Any, Iterator


def extract_bbox_xyxy(det: dict) -> Optional[List[float]]:
    """
    Extract bbox as [x1, y1, x2, y2] from a detection dict.
    Supports 'bbox', 'box', and 'x_min/y_min/x_max/y_max'.
    Returns None if invalid.
    """
    bbox = None
    if "bbox" in det:
        bbox = det["bbox"]
    elif "box" in det:
        bbox = det["box"]
    elif "x_min" in det:
        bbox = [det["x_min"], det["y_min"], det["x_max"], det["y_max"]]

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def iter_frame_entries(det_data) -> Iterator[dict]:
    """
    Iterate over per-frame entries from a detections JSON.
    Supports:
      - list of frame dicts
      - dict with 'frames' key
    """
    if isinstance(det_data, list):
        yield from det_data
    elif isinstance(det_data, dict):
        frames = det_data.get("frames", [])
        yield from frames
    else:
        return


def load_team_map(team_data) -> Dict[int, int]:
    """
    Load team_id mapping from team assignment JSON.
    Supports:
      - list of entries
      - dict with 'team_assignments', 'players', or 'tracklets'
    Returns: {track_id: team_id}
    """
    team_map = {}
    entries = []
    if isinstance(team_data, list):
        entries = team_data
    elif isinstance(team_data, dict):
        entries = team_data.get("team_assignments",
                    team_data.get("players",
                        team_data.get("tracklets", [])))
    for item in entries:
        tid = item.get("track_id")
        if tid is not None:
            team_map[tid] = item.get("team_id", -1)
    return team_map
