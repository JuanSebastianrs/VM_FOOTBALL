"""
Visual QA overlay for jersey identification.
Creates annotated frames with track_id, team_id, predicted jersey, confidence, and state.
Also exports locked error crops and locked correct crops for inspection.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.identity.schema_utils import (
    extract_bbox_xyxy,
    iter_frame_entries,
    load_team_map,
)


def load_detections(det_json, team_json):
    """Load detections and merge with team assignments."""
    with open(det_json) as f:
        det_data = json.load(f)
    with open(team_json) as f:
        team_data = json.load(f)

    team_lookup = load_team_map(team_data)

    by_frame = defaultdict(list)
    for frame_data in iter_frame_entries(det_data):
        frame_id = frame_data.get("frame_id")
        if frame_id is None:
            continue
        for det in frame_data.get("players", frame_data.get("player_detections", [])):
            tid = det.get("track_id")
            if tid is None:
                continue
            bbox = extract_bbox_xyxy(det)
            if bbox is None:
                continue
            by_frame[frame_id].append({
                "track_id": tid,
                "team_id": team_lookup.get(tid, -1),
                "bbox": bbox,
            })
    return by_frame


def load_jersey_predictions(jersey_json):
    """Load jersey predictions -> dict by track_id."""
    with open(jersey_json) as f:
        data = json.load(f)
    lookup = {}
    for t in data.get("tracklets", []):
        tid = t.get("track_id")
        lookup[tid] = {
            "predicted_number": t.get("predicted_number"),
            "confidence": t.get("confidence", 0.0),
            "state": t.get("state", "unknown"),
            "alternatives": t.get("alternatives", []),
        }
    return lookup


# Color scheme
TEAM_COLORS = {
    0: (0, 180, 255),   # orange
    1: (255, 100, 100),  # light blue
    -1: (150, 150, 150), # gray
}
STATE_COLORS = {
    "locked": (0, 255, 0),     # green
    "tentative": (0, 200, 255), # yellow
    "unknown": (100, 100, 100), # dark gray
}


def draw_overlay(frame, detections, jersey_lookup):
    """Draw detection boxes with jersey info overlay."""
    for det in detections:
        tid = det["track_id"]
        team_id = det["team_id"]
        x1, y1, x2, y2 = [int(c) for c in det["bbox"]]

        jersey = jersey_lookup.get(tid, {})
        pred_num = jersey.get("predicted_number")
        conf = jersey.get("confidence", 0.0)
        state = jersey.get("state", "unknown")

        # Box color by team
        box_color = TEAM_COLORS.get(team_id, (150, 150, 150))
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # State indicator (small dot)
        state_color = STATE_COLORS.get(state, (100, 100, 100))
        cv2.circle(frame, (x2 - 6, y1 + 6), 5, state_color, -1)

        # Label
        if pred_num is not None:
            label = f"#{pred_num}"
            conf_str = f"{conf:.0%}"
        else:
            label = "#?"
            conf_str = ""

        # Background for label
        label_text = f"T{tid} {label} {conf_str} [{state[0].upper()}]"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), box_color, -1)
        cv2.putText(frame, label_text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


def export_crops(frame, detections, jersey_lookup, frame_id, output_dir, gt_lookup=None, per_frame_gt_lookup=None):
    """Export crops for locked tracks, organized by correct/incorrect.

    Uses per-frame GT lookup when available (handles ID switches);
    falls back to track-level lookup otherwise.
    """
    for det in detections:
        tid = det["track_id"]
        jersey = jersey_lookup.get(tid, {})
        state = jersey.get("state", "unknown")

        if state != "locked":
            continue

        x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        pred_num = jersey.get("predicted_number")
        conf = jersey.get("confidence", 0.0)

        # Prefer per-frame match (handles ID switches).
        # When per-frame matches exist, do NOT fallback to track-level to avoid
        # mislabeling frames with ID switches.
        gt_num = None
        if per_frame_gt_lookup is not None:
            gt_num = per_frame_gt_lookup.get((frame_id, tid))
        elif gt_lookup is not None:
            gt_num = gt_lookup.get(tid)

        if gt_num is not None:
            correct = pred_num == gt_num
            subdir = "correct" if correct else "incorrect"
            fname = f"frame{frame_id:04d}_track{tid}_pred{pred_num}_gt{gt_num}_conf{conf:.2f}.jpg"
        else:
            subdir = "no_gt"
            fname = f"frame{frame_id:04d}_track{tid}_pred{pred_num}_conf{conf:.2f}.jpg"

        out_path = os.path.join(output_dir, subdir, fname)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, crop)


def main():
    parser = argparse.ArgumentParser(description="Jersey overlay visual QA")
    parser.add_argument("--sequence_dir", type=str, required=True)
    parser.add_argument("--detections_json", type=str, required=True)
    parser.add_argument("--team_assignments_json", type=str, required=True)
    parser.add_argument("--jersey_json", type=str, required=True)
    parser.add_argument("--output_video", type=str, required=True)
    parser.add_argument("--output_crops", type=str, default=None,
                        help="Directory to save locked crops (correct/incorrect)")
    parser.add_argument("--gt_json", type=str, default=None,
                        help="GT tracklets.json for labeling crops as correct/incorrect (dataset-native)")
    parser.add_argument("--e2e_matches_json", type=str, default=None,
                        help="End-to-end evaluation JSON (from evaluate_jersey_e2e.py) for IoU-based correctness")
    parser.add_argument("--sequence", type=str, default=None,
                        help="Sequence name for GT filtering")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--sample_every", type=int, default=5,
                        help="Export crops every N frames to avoid duplicates")
    args = parser.parse_args()

    img_dir = os.path.join(args.sequence_dir, "img1")
    frames_list = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
    print(f"Sequence: {len(frames_list)} frames")

    by_frame = load_detections(args.detections_json, args.team_assignments_json)
    jersey_lookup = load_jersey_predictions(args.jersey_json)
    print(f"Jersey predictions: {len(jersey_lookup)} tracklets")

    # Build per-frame GT lookup for precise crop labeling (handles ID switches)
    gt_lookup = None
    per_frame_gt_lookup = None  # {(frame_id, pred_track_id): gt_jersey}
    if args.e2e_matches_json:
        with open(args.e2e_matches_json) as f:
            e2e_data = json.load(f)
        # Track-level fallback (for non-per-frame JSONs)
        e2e_lookup = {}
        matches = e2e_data.get("fragment_results", e2e_data.get("matches", []))
        for m in matches:
            pred_tid = m.get("pred_track_id")
            gt_jersey = m.get("gt_jersey")
            if pred_tid is not None and gt_jersey is not None and gt_jersey > 0:
                e2e_lookup[pred_tid] = gt_jersey
        gt_lookup = e2e_lookup
        # Frame-level lookup (preferred, handles ID switches)
        per_frame_gt_lookup = {}
        for m in e2e_data.get("per_frame_matches", []):
            fid = m.get("frame_id")
            ptid = m.get("pred_track_id")
            gj = m.get("gt_jersey")
            if fid is not None and ptid is not None and gj is not None and gj > 0:
                per_frame_gt_lookup[(fid, ptid)] = gj
        print(f"E2E matches loaded: {len(e2e_lookup)} tracklets, {len(per_frame_gt_lookup)} frame-level matches")
    elif args.gt_json:
        seq_name = args.sequence or Path(args.sequence_dir).name
        with open(args.gt_json) as f:
            all_gt = json.load(f)
        gt_lookup = {}
        for r in all_gt:
            if r.get("sequence") != seq_name:
                continue
            jn = r.get("jersey_number", -1)
            try:
                jn = int(jn)
            except (ValueError, TypeError):
                continue
            if jn > 0:
                gt_lookup[r["track_id"]] = jn
        print(f"GT loaded: {len(gt_lookup)} tracklets for {seq_name}")

    # Clean crop output dirs to avoid stale files from previous runs
    if args.output_crops:
        for subdir in ["correct", "incorrect", "no_gt"]:
            path = os.path.join(args.output_crops, subdir)
            if os.path.exists(path):
                import shutil
                shutil.rmtree(path)

    # Video writer
    sample_frame = cv2.imread(os.path.join(img_dir, frames_list[0]))
    h, w = sample_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output_video, fourcc, args.fps, (w, h))

    # Stats
    total_locked = 0
    total_correct = 0
    total_incorrect = 0

    for i, fname in enumerate(frames_list):
        frame_id = i + 1
        raw_frame = cv2.imread(os.path.join(img_dir, fname))
        dets = by_frame.get(frame_id, [])

        # Export crops from RAW frame (before overlays) for clean auditing
        if args.output_crops and frame_id % args.sample_every == 0:
            export_crops(raw_frame.copy(), dets, jersey_lookup, frame_id, args.output_crops, gt_lookup, per_frame_gt_lookup)

        # Draw overlays for video
        frame = draw_overlay(raw_frame, dets, jersey_lookup)

        # Add legend
        cv2.putText(frame, f"Frame {frame_id}/{len(frames_list)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "[L]ocked  [T]entative  [U]nknown", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(frame)

        if i % 100 == 0:
            print(f"  {frame_id}/{len(frames_list)}...")

    writer.release()
    print(f"\nVideo saved: {args.output_video}")

    # Summary stats from crops
    if args.output_crops:
        for subdir in ["correct", "incorrect", "no_gt"]:
            path = os.path.join(args.output_crops, subdir)
            if os.path.exists(path):
                n = len(os.listdir(path))
                print(f"  {subdir}: {n} crops")


if __name__ == "__main__":
    main()
