"""
Visual Validator: Overlay bbox detections from JSON onto original frame sequences.
Creates side-by-side comparison video of ByteTrack vs ASAHI detections.

Usage:
    python cloud/ball_detection/validation/visualize_detections.py
"""

import cv2
import json
import os
import glob
import sys

# ---- Configuration ----
FRAMES_DIR = os.path.join("datasets", "test_seq_116")
VALIDATION_DIR = os.path.join("cloud", "ball_detection", "validation")
BYTETRACK_JSON = os.path.join(VALIDATION_DIR, "bytetrack_coordinates.json")
ASAHI_JSON = os.path.join(VALIDATION_DIR, "asahi_coordinates.json")

OUTPUT_BYTETRACK = os.path.join(VALIDATION_DIR, "bytetrack_validation.mp4")
OUTPUT_ASAHI = os.path.join(VALIDATION_DIR, "asahi_validation.mp4")
OUTPUT_SIDEBYSIDE = os.path.join(VALIDATION_DIR, "comparison_sidebyside.mp4")

# Colors
COLOR_BT = (0, 200, 255)       # Orange-yellow for ByteTrack
COLOR_SAHI_HIGH = (0, 255, 0)   # Green for SAHI conf >= 0.5
COLOR_SAHI_LOW = (0, 255, 255)  # Yellow for SAHI conf < 0.5
COLOR_FP_RISK = (0, 0, 255)    # Red for conf < 0.25 (likely false positive)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def draw_detections(frame, detections, mode="bytetrack"):
    """Draw bboxes on frame with color-coded confidence."""
    annotated = frame.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["conf"]
        source = det.get("source", mode)
        track_id = det.get("track_id", "")
        
        # Color code by confidence
        if conf < 0.25:
            color = COLOR_FP_RISK  # Red = low conf, probable false positive
            label_prefix = "FP?"
        elif mode == "bytetrack":
            color = COLOR_BT
            label_prefix = f"T{track_id}" if track_id != "" else "BT"
        elif conf >= 0.5:
            color = COLOR_SAHI_HIGH
            label_prefix = "SAHI"
        else:
            color = COLOR_SAHI_LOW
            label_prefix = "SAHI"
        
        # Draw bbox
        thickness = 2 if conf >= 0.5 else 1
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        label = f"{label_prefix} {conf:.2f}"
        font_scale = 0.45
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)
    
    return annotated


def add_stats_overlay(frame, frame_name, detections, mode, frame_idx, total_frames):
    """Add stats bar at the top."""
    h, w = frame.shape[:2]
    
    # Dark overlay bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    n_det = len(detections)
    confs = [d["conf"] for d in detections] if detections else [0]
    max_conf = max(confs)
    n_fp = sum(1 for d in detections if d["conf"] < 0.25)
    
    text = f"{mode.upper()} | Frame {frame_idx+1}/{total_frames} | {frame_name} | Det: {n_det} | MaxConf: {max_conf:.2f}"
    if n_fp > 0:
        text += f" | FP_RISK: {n_fp}"
    
    color = (0, 200, 255) if mode == "bytetrack" else (0, 255, 0)
    cv2.putText(frame, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    
    return frame


def generate_video(frames_dir, json_path, output_path, mode, fps=25):
    """Generate annotated video for a single pipeline."""
    data = load_json(json_path)
    image_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    
    if not image_files:
        print(f"[ERROR] No images in {frames_dir}")
        return None
    
    first = cv2.imread(image_files[0])
    h, w = first.shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    total_frames = len(image_files)
    total_detections = 0
    total_fp_risk = 0
    frames_with_det = 0
    empty_frames = 0
    
    print(f"\n[{mode.upper()}] Generating validation video...")
    
    for i, img_path in enumerate(image_files):
        frame = cv2.imread(img_path)
        frame_name = os.path.basename(img_path)
        
        dets = data.get(frame_name, [])
        total_detections += len(dets)
        if dets:
            frames_with_det += 1
        else:
            empty_frames += 1
        
        fp_risk = [d for d in dets if d["conf"] < 0.25]
        total_fp_risk += len(fp_risk)
        
        annotated = draw_detections(frame, dets, mode)
        annotated = add_stats_overlay(annotated, frame_name, dets, mode, i, total_frames)
        
        out.write(annotated)
    
    out.release()
    
    # Print summary
    print(f"  Total detections: {total_detections}")
    print(f"  Frames with detections: {frames_with_det}/{total_frames} ({frames_with_det/total_frames*100:.1f}%)")
    print(f"  Empty frames (no ball found): {empty_frames}")
    print(f"  FP risk (conf < 0.25): {total_fp_risk} ({total_fp_risk/max(total_detections,1)*100:.1f}%)")
    print(f"  Video saved: {output_path}")
    
    return output_path


def generate_sidebyside(frames_dir, bt_json, sahi_json, output_path, fps=25):
    """Generate side-by-side comparison video (ByteTrack left, ASAHI right)."""
    bt_data = load_json(bt_json)
    sahi_data = load_json(sahi_json)
    image_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    
    if not image_files:
        print("[ERROR] No images found")
        return None
    
    first = cv2.imread(image_files[0])
    h, w = first.shape[:2]
    
    # Side by side: half width each
    half_w = w // 2
    canvas_w = half_w * 2
    canvas_h = h // 2 + 30  # scaled down + label bar
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (canvas_w, canvas_h))
    
    print(f"\n[SIDE-BY-SIDE] Generating comparison video...")
    
    for i, img_path in enumerate(image_files):
        frame = cv2.imread(img_path)
        frame_name = os.path.basename(img_path)
        
        bt_dets = bt_data.get(frame_name, [])
        sahi_dets = sahi_data.get(frame_name, [])
        
        # Draw detections
        bt_frame = draw_detections(frame, bt_dets, "bytetrack")
        sahi_frame = draw_detections(frame, sahi_dets, "asahi")
        
        # Scale down to half
        bt_small = cv2.resize(bt_frame, (half_w, h // 2))
        sahi_small = cv2.resize(sahi_frame, (half_w, h // 2))
        
        # Create canvas
        canvas = cv2.copyMakeBorder(
            cv2.hconcat([bt_small, sahi_small]),
            30, 0, 0, 0,
            cv2.BORDER_CONSTANT, value=(30, 30, 30)
        )
        
        # Labels
        bt_label = f"BYTETRACK ({len(bt_dets)} det)"
        sahi_label = f"ASAHI ({len(sahi_dets)} det)"
        frame_label = f"Frame {i+1}/{len(image_files)}: {frame_name}"
        
        cv2.putText(canvas, bt_label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_BT, 1)
        cv2.putText(canvas, sahi_label, (half_w + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_SAHI_HIGH, 1)
        cv2.putText(canvas, frame_label, (canvas_w // 2 - 100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        # Divider line
        cv2.line(canvas, (half_w, 30), (half_w, canvas_h), (100, 100, 100), 1)
        
        out.write(canvas)
    
    out.release()
    print(f"  Comparison video saved: {output_path}")
    return output_path


def analyze_false_positives(json_path, mode):
    """Analyze detections for potential false positives."""
    data = load_json(json_path)
    
    low_conf = []  # conf < 0.25
    multi_det = []  # frames with > 1 detection (should be rare for ball)
    
    for frame_name, dets in data.items():
        for d in dets:
            if d["conf"] < 0.25:
                low_conf.append({"frame": frame_name, **d})
        if len(dets) > 1:
            multi_det.append({"frame": frame_name, "count": len(dets), 
                             "confs": [d["conf"] for d in dets]})
    
    print(f"\n{'='*50}")
    print(f" FALSE POSITIVE ANALYSIS: {mode.upper()}")
    print(f"{'='*50}")
    print(f"  Low confidence detections (conf < 0.25): {len(low_conf)}")
    
    if low_conf:
        print(f"  Examples:")
        for fp in low_conf[:5]:
            print(f"    {fp['frame']}: bbox={fp['bbox']} conf={fp['conf']:.3f}")
    
    print(f"\n  Multi-detection frames (>1 ball): {len(multi_det)}")
    if multi_det:
        print(f"  ⚠️  These are FP candidates (only 1 ball in play):")
        for md in multi_det[:10]:
            print(f"    {md['frame']}: {md['count']} dets, confs={[round(c,2) for c in md['confs']]}")
    
    print()
    return low_conf, multi_det


if __name__ == "__main__":
    print("=" * 60)
    print(" SAM2 Detection Validator")
    print("=" * 60)
    
    # Check inputs exist
    if not os.path.exists(FRAMES_DIR):
        print(f"[ERROR] Frames directory not found: {FRAMES_DIR}")
        sys.exit(1)
    
    for jf in [BYTETRACK_JSON, ASAHI_JSON]:
        if not os.path.exists(jf):
            print(f"[ERROR] JSON not found: {jf}")
            sys.exit(1)
    
    # 1. Analyze false positive risk
    bt_fp, bt_multi = analyze_false_positives(BYTETRACK_JSON, "ByteTrack")
    sahi_fp, sahi_multi = analyze_false_positives(ASAHI_JSON, "ASAHI")
    
    # 2. Generate individual annotated videos
    generate_video(FRAMES_DIR, BYTETRACK_JSON, OUTPUT_BYTETRACK, "bytetrack")
    generate_video(FRAMES_DIR, ASAHI_JSON, OUTPUT_ASAHI, "asahi")
    
    # 3. Generate side-by-side comparison
    generate_sidebyside(FRAMES_DIR, BYTETRACK_JSON, ASAHI_JSON, OUTPUT_SIDEBYSIDE)
    
    print("\n" + "=" * 60)
    print(" DONE! Open these files to validate:")
    print("=" * 60)
    print(f"  1. {OUTPUT_BYTETRACK}")
    print(f"  2. {OUTPUT_ASAHI}")
    print(f"  3. {OUTPUT_SIDEBYSIDE}  ← recommended")
    print()
    print("  Color code:")
    print("    🟢 Green    = SAHI detection, conf >= 0.5")
    print("    🟡 Yellow   = SAHI detection, conf < 0.5")
    print("    🟠 Orange   = ByteTrack detection")
    print("    🔴 Red      = conf < 0.25 (possible false positive)")
