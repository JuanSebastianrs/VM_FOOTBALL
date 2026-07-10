"""
Per-tracklet appearance embeddings for offline tracklet linking.

For every sequence with cached detections, samples up to N crops per tracklet
(evenly spread over its lifespan), embeds them with an OSNet person-ReID model
and stores the L2-normalized mean embedding per tracklet:

    outputs/<seq>/<seq>_tracklet_embeddings.npz   (tids: int64[K], emb: float32[K, D])

Each frame image is read once; all tracklet crops of that frame are batched.

Usage:
    $env:PYTHONPATH="."
    python core/tracking/extract_tracklet_embeddings.py [--sequences SNMOT-116 ...]
"""

import os
import sys
import json
import glob
import argparse
import collections

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.tracking.offline_tracklet_linker import load_tracklets  # noqa: E402

DATA_ROOT = r"data\tracking\SoccerNet\tracking\test\test"
PREDS_DIR = "outputs"
MODEL_PATH = r"models\reid\osnet_x1_0_msmt17.pth"
SAMPLES_PER_TRACKLET = 12
BATCH_SIZE = 128
CROP_MARGIN = 0.05  # fraction of box size added on each side


def sample_frames(tracklet, n):
    if len(tracklet.frames) <= n:
        return list(tracklet.frames)
    idx = np.linspace(0, len(tracklet.frames) - 1, n).round().astype(int)
    return [tracklet.frames[i] for i in sorted(set(idx))]


def process_sequence(seq, extractor):
    seq_dir = os.path.join(DATA_ROOT, seq, "img1")
    det_path = os.path.join(PREDS_DIR, seq, f"{seq}_detections.json")
    out_path = os.path.join(PREDS_DIR, seq, f"{seq}_tracklet_embeddings.npz")
    if os.path.exists(out_path):
        print(f"  [EMB] {seq}: already done, skipping")
        return
    if not (os.path.isdir(seq_dir) and os.path.exists(det_path)):
        print(f"  [EMB] {seq}: missing images or detections, skipping")
        return

    with open(det_path) as f:
        tracklets = load_tracklets(json.load(f))

    # frame -> [(tid, box), ...] so each image is decoded once
    per_frame = collections.defaultdict(list)
    for tr in tracklets:
        for f in sample_frames(tr, SAMPLES_PER_TRACKLET):
            per_frame[f].append((tr.tid, tr.boxes[f]))

    crops, crop_tids = [], []
    for frame in sorted(per_frame):
        img_path = os.path.join(seq_dir, f"{frame:06d}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        for tid, (x, y, w, h) in per_frame[frame]:
            mx, my = w * CROP_MARGIN, h * CROP_MARGIN
            x0, y0 = max(int(x - mx), 0), max(int(y - my), 0)
            x1, y1 = min(int(x + w + mx), W), min(int(y + h + my), H)
            if x1 - x0 < 8 or y1 - y0 < 16:
                continue
            crops.append(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
            crop_tids.append(tid)

    if not crops:
        print(f"  [EMB] {seq}: no usable crops")
        return

    feats = []
    for i in range(0, len(crops), BATCH_SIZE):
        feats.append(extractor(crops[i:i + BATCH_SIZE]).cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True).clip(min=1e-8)

    by_tid = collections.defaultdict(list)
    for tid, f in zip(crop_tids, feats):
        by_tid[tid].append(f)
    tids = np.array(sorted(by_tid), dtype=np.int64)
    emb = np.stack([np.mean(by_tid[t], axis=0) for t in tids]).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-8)

    np.savez(out_path, tids=tids, emb=emb)
    print(f"  [EMB] {seq}: {len(tids)} tracklets, {len(crops)} crops -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from torchreid.reid.utils import FeatureExtractor
    extractor = FeatureExtractor(
        model_name="osnet_x1_0", model_path=MODEL_PATH, device=args.device
    )

    seqs = args.sequences or sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(PREDS_DIR, "SNMOT-*"))
        if os.path.exists(os.path.join(p, f"{os.path.basename(p)}_detections.json"))
    )
    print(f"Extracting embeddings for {len(seqs)} sequences...")
    for seq in seqs:
        process_sequence(seq, extractor)
    print("DONE")


if __name__ == "__main__":
    main()
