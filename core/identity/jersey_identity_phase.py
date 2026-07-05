"""
Jersey Identity Phase: inference CLI for tracklet-level jersey number recognition.
Supports two inference modes:
  - "mil"   : Multi-Instance Learning on top-K crops (legacy, fast).
  - "temporal" : Per-frame inference on ALL frames + temporal fusion + peak detection (robust).

Usage:
    python core/identity/jersey_identity_phase.py \
        --sequence_dir data/tracking/SoccerNet/tracking/test/test/SNMOT-148 \
        --detections_json outputs/SNMOT-148/SNMOT-148_detections.json \
        --team_assignments_json outputs/SNMOT-148/SNMOT-148_team_assignments.json \
        --model_path runs/jersey_digit_mil_v2/best.pt \
        --output_json outputs/SNMOT-148/SNMOT-148_jersey_identity.json \
        --inference_mode temporal \
        --device cuda:0
"""

import sys
import argparse
import json
import random
import math
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.identity.jersey_model import (
    TRANSFORM_INFERENCE,
    get_model_transform,
    load_jersey_model,
    compute_jersey_probs_from_logits,
    load_legibility_model,
)
from core.identity.jersey_assignment import TrackletInfo, assign_per_team
from core.identity.tracklet_linking import link_fragments, detect_mode_switch
from core.identity.schema_utils import (
    extract_bbox_xyxy,
    iter_frame_entries,
    load_team_map,
)


# ──────────────────────────────────────────────────────────────────────
# Crop extraction
# ──────────────────────────────────────────────────────────────────────

from core.identity.crops import (
    extract_torso_crop_from_image,
    extract_candidate_crops_from_image,
    compute_crop_quality_rgb,
    compute_crop_quality,
)


# ──────────────────────────────────────────────────────────────────────
# Tracklet building
# ──────────────────────────────────────────────────────────────────────

def build_tracklet_frames(detections_json, team_assignments_json, sequence_dir,
                          K=16, min_frames=8, min_quality=0.0, min_bbox_area=400):
    """
    Reconstruct tracklet frame bags from detections + team assignments.
    Returns list of tracklet dicts with ALL frames (for temporal) and top-K frames (for MIL).
    """
    with open(detections_json) as f:
        det_data = json.load(f)
    with open(team_assignments_json) as f:
        team_data = json.load(f)

    sequence_dir = Path(sequence_dir)
    img1_dir = sequence_dir / "img1"

    team_map = load_team_map(team_data)

    # Store ALL frames per tracklet
    tracklet_all = defaultdict(lambda: {
        "frame_ids": [], "bboxes": [], "quality_scores": [], "team_id": -1
    })

    for frame_entry in iter_frame_entries(det_data):
        fid = frame_entry.get("frame_id")
        if fid is None:
            continue
        players = frame_entry.get("players", frame_entry.get("player_detections", []))
        for d in players:
            tid = d.get("track_id")
            if tid is None:
                continue
            bbox = extract_bbox_xyxy(d)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            if area < min_bbox_area:
                continue

            frame_path = img1_dir / f"{fid:06d}.jpg"
            if not frame_path.exists():
                continue
            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            crop = extract_torso_crop_from_image(img, bbox)
            quality = compute_crop_quality(crop)

            tracklet_all[tid]["frame_ids"].append(fid)
            tracklet_all[tid]["bboxes"].append([x1, y1, x2, y2])
            tracklet_all[tid]["quality_scores"].append(quality)
            tracklet_all[tid]["team_id"] = team_map.get(tid, -1)

    tracklets = []
    for tid, data in tracklet_all.items():
        # Filter to frames passing min_quality
        valid_idx = [i for i, q in enumerate(data["quality_scores"]) if q >= min_quality]
        if len(valid_idx) < min_frames:
            continue

        valid_frame_ids = [data["frame_ids"][i] for i in valid_idx]
        valid_bboxes = [data["bboxes"][i] for i in valid_idx]
        valid_qualities = [data["quality_scores"][i] for i in valid_idx]

        # Top-K by quality
        sorted_idx = sorted(range(len(valid_qualities)), key=lambda i: valid_qualities[i], reverse=True)
        top_idx = sorted_idx[:K]

        tracklets.append({
            "track_id": tid,
            "team_id": data["team_id"],
            "frame_ids": [valid_frame_ids[i] for i in top_idx],
            "bboxes": [valid_bboxes[i] for i in top_idx],
            "quality_scores": [valid_qualities[i] for i in top_idx],
            "all_frame_ids": valid_frame_ids,
            "all_bboxes": valid_bboxes,
            "all_quality_scores": valid_qualities,
            "num_frames": len(valid_idx),
        })
    return tracklets


# ──────────────────────────────────────────────────────────────────────
# MIL inference (legacy)
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_tracklets_mil(model, tracklets, sequence_dir, device, K=16, seed=42):
    """Run MIL inference on top-K crops per tracklet."""
    model.eval()
    digit_tf = get_model_transform(model)
    img1_dir = Path(sequence_dir) / "img1"
    results = {}

    for t in tracklets:
        tid = t["track_id"]
        frame_ids = t["frame_ids"][:K]
        crops = []
        for fid in frame_ids:
            fp = img1_dir / f"{fid:06d}.jpg"
            if not fp.exists():
                continue
            img = cv2.imread(str(fp))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bbox = t["bboxes"][t["frame_ids"].index(fid)]
            crop = extract_torso_crop_from_image(img, bbox)
            if crop is None:
                continue
            crops.append(digit_tf(crop))

        if len(crops) == 0:
            results[tid] = {"probs": None, "alternatives": [], "mode": "mil"}
            continue

        rng = random.Random(seed + tid)
        while len(crops) < K:
            crops.append(crops[rng.randrange(len(crops))])

        x = torch.stack(crops[:K]).unsqueeze(0).to(device)
        out_len, out_tens, out_ones = model(x)
        jersey_probs = compute_jersey_probs_from_logits(out_len, out_tens, out_ones)

        topk_idx = np.argsort(jersey_probs)[::-1][:5]
        alternatives = [(int(idx + 1), float(jersey_probs[idx])) for idx in topk_idx]

        results[tid] = {"probs": jersey_probs, "alternatives": alternatives, "mode": "mil"}

    return results


# ──────────────────────────────────────────────────────────────────────
# Per-frame inference + temporal fusion
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _parseq_read(parseq, parseq_tf, crop_rgb, device):
    """Lee el crop con PARSeq. Devuelve (numero 1..99 | None, confianza)."""
    from PIL import Image as _Image
    x = parseq_tf(_Image.fromarray(crop_rgb)).unsqueeze(0).to(device)
    logits = parseq(x)
    labels, confs = parseq.tokenizer.decode(logits.softmax(-1))
    digits = "".join(ch for ch in labels[0] if ch.isdigit())
    if not digits or len(digits) > 2:
        return None, 0.0
    n = int(digits)
    if not (1 <= n <= 99):
        return None, 0.0
    c = confs[0]
    conf = float(c.prod().clamp(0, 1)) if hasattr(c, "prod") else float(c)
    return n, conf


def load_parseq(model_name, device, checkpoint=None):
    """Carga PARSeq (torch.hub) + su transform oficial.

    checkpoint: ruta opcional a un .pt de finetune_parseq_jersey.py
    (dict con 'state_dict'); se aplica sobre la arquitectura del hub.
    """
    parseq = torch.hub.load("baudm/parseq", model_name, pretrained=True,
                            trust_repo=True)
    if checkpoint:
        from pathlib import Path as _Path
        if _Path(checkpoint).exists():
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)
            parseq.load_state_dict(state)
            print(f"  PARSeq fine-tuned weights: {checkpoint} "
                  f"(val {ckpt.get('val_acc', float('nan')):.3f})")
        else:
            print(f"  AVISO: checkpoint PARSeq no encontrado ({checkpoint}); "
                  f"se usan los pesos genericos del hub")
    parseq = parseq.eval().to(device)
    try:
        from strhub.data.module import SceneTextDataModule
        tf = SceneTextDataModule.get_transform(parseq.hparams.img_size)
    except Exception:
        import torchvision.transforms as T
        tf = T.Compose([T.Resize((32, 128)), T.ToTensor(), T.Normalize(0.5, 0.5)])
    return parseq, tf


def predict_per_frame(model, tracklet, sequence_dir, device, legibility_model=None, legibility_threshold=0.5, multi_crop=False,
                      parseq=None, parseq_tf=None, parseq_weight=0.25):
    """Run inference on EVERY frame of a tracklet. Returns list of frame results."""
    model.eval()
    if legibility_model is not None:
        legibility_model.eval()
    # Digit model input size comes from the checkpoint (224 for v1.7+);
    # the legibility model always runs at 128 (TRANSFORM_INFERENCE).
    digit_tf = get_model_transform(model)
    img1_dir = Path(sequence_dir) / "img1"
    frame_results = []

    for fid, bbox, quality in zip(tracklet["all_frame_ids"],
                                   tracklet["all_bboxes"],
                                   tracklet["all_quality_scores"]):
        fp = img1_dir / f"{fid:06d}.jpg"
        if not fp.exists():
            continue
        img = cv2.imread(str(fp))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Standard torso crop as base fallback
        standard_crop = extract_torso_crop_from_image(img, bbox)
        if standard_crop is None:
            continue

        selected_crop = standard_crop
        selected_quality = quality
        selected_legibility = 1.0
        selected_variant = "torso_actual"

        if legibility_model is not None:
            if multi_crop and quality >= 0.2:
                # Extract candidate crops
                candidates = extract_candidate_crops_from_image(img, bbox)
                
                # Filter out None crops and compute their CPU qualities
                valid_candidates = []
                valid_crops = []
                for cand in candidates:
                    crop_img = cand["crop"]
                    if crop_img is not None:
                        q_score = compute_crop_quality_rgb(crop_img)
                        valid_candidates.append({
                            "name": cand["name"],
                            "crop": crop_img,
                            "quality": q_score
                        })
                        valid_crops.append(crop_img)

                if valid_crops:
                    # Batch evaluate legibility
                    batch_tensors = [TRANSFORM_INFERENCE(c) for c in valid_crops]
                    batch_tensor = torch.stack(batch_tensors).to(device)
                    logits = legibility_model(batch_tensor)
                    legibilities = torch.sigmoid(logits).cpu().numpy()

                    # Find best crop maximizing quality * legibility
                    best_idx = -1
                    best_score = -1.0
                    for idx, cand in enumerate(valid_candidates):
                        score = cand["quality"] * legibilities[idx]
                        if score > best_score:
                            best_score = score
                            best_idx = idx

                    if best_idx != -1:
                        best_cand = valid_candidates[best_idx]
                        best_legibility = legibilities[best_idx]
                        
                        if best_legibility >= legibility_threshold:
                            selected_crop = best_cand["crop"]
                            selected_quality = best_cand["quality"]
                            selected_legibility = float(best_legibility)
                            selected_variant = best_cand["name"]
                        else:
                            # Fallback to standard torso crop
                            std_idx = next((i for i, c in enumerate(valid_candidates) if c["name"] == "torso_actual"), -1)
                            if std_idx != -1:
                                selected_legibility = float(legibilities[std_idx])
                            else:
                                std_tensor = TRANSFORM_INFERENCE(standard_crop).unsqueeze(0).to(device)
                                std_logit = legibility_model(std_tensor)
                                selected_legibility = float(torch.sigmoid(std_logit).item())
                            selected_crop = standard_crop
                            selected_quality = quality
                            selected_variant = "torso_actual"
                else:
                    std_tensor = TRANSFORM_INFERENCE(standard_crop).unsqueeze(0).to(device)
                    std_logit = legibility_model(std_tensor)
                    selected_legibility = float(torch.sigmoid(std_logit).item())
                    selected_crop = standard_crop
                    selected_quality = quality
                    selected_variant = "torso_actual"
            else:
                # Single crop evaluate legibility
                std_tensor = TRANSFORM_INFERENCE(standard_crop).unsqueeze(0).to(device)
                std_logit = legibility_model(std_tensor)
                selected_legibility = float(torch.sigmoid(std_logit).item())
                selected_crop = standard_crop
                selected_quality = quality
                selected_variant = "torso_actual"
        else:
            # No legibility model
            selected_crop = standard_crop
            selected_quality = quality
            selected_legibility = 1.0
            selected_variant = "torso_actual"

        crop_tensor = digit_tf(selected_crop)
        x = crop_tensor.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 3, H, W)
        out_len, out_tens, out_ones = model(x)
        jersey_probs = compute_jersey_probs_from_logits(out_len, out_tens, out_ones)

        # Ensamble opcional con PARSeq (segundo lector de texto de escena):
        # solo cuando PARSeq LEE un numero valido en un crop legible.
        if parseq is not None and selected_legibility >= 0.30:
            num, conf = _parseq_read(parseq, parseq_tf, selected_crop, device)
            if num is not None and conf > 0:
                pp = np.full(99, (1.0 - conf) / 98.0)
                pp[num - 1] = conf
                w = parseq_weight
                mix = (np.power(jersey_probs + 1e-12, 1.0 - w)
                       * np.power(pp + 1e-12, w))
                jersey_probs = mix / mix.sum()

        frame_results.append({
            "frame_id": int(fid),
            "jersey_probs": jersey_probs,
            "quality_score": float(selected_quality),
            "legibility_score": float(selected_legibility),
            "crop_variant": str(selected_variant),
            "crop_area": int(selected_crop.shape[0] * selected_crop.shape[1]),
        })

    return frame_results


def temporal_fusion(frame_results, p1_threshold=0.85, margin_threshold=0.20,
                    roster_set=None, min_peak_quality=0.3,
                    fusion_mode="geometric", temperature=1.0, topk_frac=0.5):
    """
    Fuse per-frame distributions into tracklet-level posterior.

    fusion_mode:
      - "geometric"      : log-space weighted fusion (default, legacy behavior).
      - "arithmetic"     : weighted average of probabilities (robust to
                           confident-wrong frames that dominate log-space).
      - "topk_geometric" : geometric fusion restricted to the top `topk_frac`
                           fraction of frames by weight (quality * legibility).
      - "confidence_topk": geometric fusion restricted to the frames donde el
                           MODELO es mas confiado (max prob por frame); los
                           frames con numero claramente visible dominan la
                           decision en vez de diluirse entre frames ilegibles.
    temperature: softens (>1) or sharpens (<1) per-frame distributions before
                 fusion; applied as p^(1/T) renormalized.
    """
    if not frame_results:
        return {
            "probs": None, "predicted_number": None, "confidence": 0.0,
            "state": "unknown", "p1": 0.0, "margin": 0.0,
            "alternatives": [], "peaks": [], "peak_scores": {},
            "num_frames_processed": 0,
        }

    probs = np.array([fr["jersey_probs"] for fr in frame_results])  # (T, 99)
    qualities = np.array([fr["quality_score"] for fr in frame_results])
    legibilities = np.array([fr.get("legibility_score", 1.0) for fr in frame_results])

    # Temperature scaling: p^(1/T), renormalized per frame
    if temperature != 1.0 and temperature > 0:
        probs = np.power(probs + 1e-12, 1.0 / temperature)
        probs = probs / probs.sum(axis=1, keepdims=True)

    # Normalize weights: weight = quality_score * legibility_score
    weights = qualities * legibilities
    weights_sum = weights.sum()
    if np.isnan(weights_sum) or weights_sum <= 1e-8:
        weights = np.ones_like(weights) / max(len(weights), 1)
    else:
        weights = weights / weights_sum

    # Optional restriction to top-weighted frames before geometric fusion
    if fusion_mode == "confidence_topk":
        conf = probs.max(axis=1)  # confianza de prediccion por frame
        if len(conf) > 2:
            k = max(2, int(np.ceil(topk_frac * len(conf))))
            idx = np.argsort(conf)[::-1][:k]
        else:
            idx = np.arange(len(conf))
        sub_w = weights[idx] * conf[idx]
        sub_w = sub_w / max(sub_w.sum(), 1e-8)
        fused_log = (sub_w[:, None] * np.log(probs[idx] + 1e-10)).sum(axis=0)
    elif fusion_mode == "topk_geometric" and len(weights) > 2:
        k = max(2, int(np.ceil(topk_frac * len(weights))))
        top_idx = np.argsort(weights)[::-1][:k]
        sub_w = weights[top_idx]
        sub_w = sub_w / max(sub_w.sum(), 1e-8)
        fused_log = (sub_w[:, None] * np.log(probs[top_idx] + 1e-10)).sum(axis=0)
    elif fusion_mode == "arithmetic":
        fused_ari = (weights[:, None] * probs).sum(axis=0)
        fused_log = np.log(fused_ari + 1e-10)
    else:  # "geometric" (legacy default)
        log_probs = np.log(probs + 1e-10)
        fused_log = (weights[:, None] * log_probs).sum(axis=0)

    # Roster mask (soft: set to -inf, not hard zero)
    if roster_set:
        for n in range(1, 100):
            if n not in roster_set:
                fused_log[n - 1] = -np.inf

    # Back to probabilities
    max_log = np.max(fused_log)
    if np.isnan(max_log) or np.isinf(max_log):
        fused_log = np.zeros_like(fused_log)
    else:
        fused_log = fused_log - max_log
        
    fused_probs = np.exp(fused_log)
    fused_probs_sum = fused_probs.sum()
    if np.isnan(fused_probs_sum) or fused_probs_sum <= 1e-8:
        fused_probs = np.ones_like(fused_probs) / len(fused_probs)
    else:
        fused_probs = fused_probs / fused_probs_sum

    # Top-1
    best_idx = int(np.argmax(fused_probs))
    best_prob = float(fused_probs[best_idx])
    best_num = best_idx + 1

    # Margin vs second-best
    sorted_probs = np.sort(fused_probs)[::-1]
    p2 = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin = best_prob - p2

    # Peak detection: local maxima in quality
    peaks = []
    for i in range(len(frame_results)):
        left = qualities[i - 1] if i > 0 else -np.inf
        right = qualities[i + 1] if i < len(qualities) - 1 else -np.inf
        if qualities[i] > left and qualities[i] > right and qualities[i] >= min_peak_quality:
            pred_idx = int(np.argmax(frame_results[i]["jersey_probs"]))
            peaks.append({
                "frame_id": int(frame_results[i]["frame_id"]),
                "quality": float(qualities[i]),
                "predicted_number": pred_idx + 1,
                "confidence": float(frame_results[i]["jersey_probs"][pred_idx]),
                "crop_variant": frame_results[i].get("crop_variant", "torso_actual"),
                "legibility_score": float(frame_results[i].get("legibility_score", 1.0)),
            })

    # Group peaks by predicted number
    peak_groups = defaultdict(list)
    for p in peaks:
        peak_groups[p["predicted_number"]].append(p)

    peak_scores = {}
    for num, group in peak_groups.items():
        peak_scores[num] = sum(g["quality"] * g["confidence"] for g in group)

    # State
    if best_prob >= p1_threshold and margin >= margin_threshold:
        state = "locked"
    elif best_prob >= 0.40 and margin >= 0.10:
        state = "tentative"
    else:
        state = "unknown"

    # Alternatives
    topk_idx = np.argsort(fused_probs)[::-1][:5]
    alternatives = [(int(idx + 1), float(fused_probs[idx])) for idx in topk_idx]

    return {
        "probs": fused_probs,
        "predicted_number": best_num if state != "unknown" else None,
        "confidence": best_prob,
        "state": state,
        "p1": best_prob,
        "margin": margin,
        "alternatives": alternatives,
        "peaks": peaks,
        "peak_groups": {int(k): v for k, v in peak_groups.items()},
        "peak_scores": {int(k): v for k, v in peak_scores.items()},
        "num_frames_processed": len(frame_results),
    }


@torch.no_grad()
def predict_tracklets_temporal(model, tracklets, sequence_dir, device,
                               p1_threshold=0.85, margin_threshold=0.20,
                               team_rosters=None, legibility_model=None,
                               legibility_threshold=0.5, multi_crop=False,
                               min_legible_frames=4, min_peak_quality=0.3,
                               fusion_mode="geometric", temperature=1.0,
                               do_link_fragments=False, split_on_switch=False,
                               min_trim_fraction=0.30,
                               parseq=None, parseq_tf=None, parseq_weight=0.25):
    """
    Run per-frame inference + temporal fusion for all tracklets.

    do_link_fragments: link ByteTrack fragments of the same player (team +
        bounded temporal gap + spatial continuity + identity compatibility of
        the per-fragment fused predictions) and pool their evidence.
    split_on_switch: detect probable ID switches inside a (pooled) tracklet.
        If the minority segment is large (>= min_trim_fraction of frames) the
        fusion uses only the dominant segment; in any detected switch the
        tracklet is never locked.
    """
    # 1) Per-frame inference + legibility filter, per fragment
    frame_results_by_tid = {}
    for t in tracklets:
        tid = t["track_id"]
        frame_results = predict_per_frame(
            model, t, sequence_dir, device,
            legibility_model=legibility_model,
            legibility_threshold=legibility_threshold,
            multi_crop=multi_crop,
            parseq=parseq, parseq_tf=parseq_tf, parseq_weight=parseq_weight,
        )
        # Filter frame results by legibility score (ignore frames < threshold, but only if >= min_legible_frames remain)
        if legibility_model is not None:
            filtered_frame_results = [fr for fr in frame_results if fr.get("legibility_score", 1.0) >= legibility_threshold]
            if len(filtered_frame_results) >= min_legible_frames:
                frame_results = filtered_frame_results
        frame_results_by_tid[tid] = frame_results

    def _fuse(frames, team_id):
        roster = team_rosters.get(team_id) if team_rosters else None
        return temporal_fusion(
            frames,
            p1_threshold=p1_threshold,
            margin_threshold=margin_threshold,
            roster_set=roster,
            min_peak_quality=min_peak_quality,
            fusion_mode=fusion_mode,
            temperature=temperature,
        )

    # 2) Optional fragment linking, gated by identity compatibility of the
    #    per-fragment fusion (never pool two confident different players)
    link_groups = None
    if do_link_fragments:
        fragment_summary = {}
        for t in tracklets:
            frag_fused = _fuse(frame_results_by_tid[t["track_id"]], t["team_id"])
            fragment_summary[t["track_id"]] = (
                frag_fused["alternatives"][0][0] if frag_fused["alternatives"] else None,
                frag_fused.get("p1", 0.0),
            )
        link_groups = link_fragments(tracklets, fragment_summary=fragment_summary)
        n_linked = sum(1 for tid, gid in link_groups.items() if gid != tid)
        print(f"  Fragment linking: {n_linked} fragments joined an identity chain")

    groups = defaultdict(list)
    for t in tracklets:
        tid = t["track_id"]
        gid = link_groups.get(tid, tid) if link_groups else tid
        groups[gid].append(t)

    # 3) Pooled fusion per identity group
    results = {}
    for gid, members in groups.items():
        pooled = []
        for m in members:
            pooled.extend(frame_results_by_tid[m["track_id"]])
        pooled.sort(key=lambda fr: fr["frame_id"])

        switch_at = None
        if split_on_switch and pooled:
            switch_at, left_mode, right_mode = detect_mode_switch(pooled)
            if switch_at is not None:
                left, right = pooled[:switch_at], pooled[switch_at:]
                minority = min(len(left), len(right)) / max(len(pooled), 1)
                if minority >= min_trim_fraction:
                    # Large contamination: fuse only the dominant segment
                    pooled = left if len(left) >= len(right) else right

        fused = _fuse(pooled, members[0]["team_id"])
        # A detected switch means contaminated identity: never lock it
        if switch_at is not None and fused.get("state") == "locked":
            fused["state"] = "tentative"

        fused["mode"] = "temporal"
        fused["linked_group"] = gid
        fused["linked_members"] = [m["track_id"] for m in members]
        fused["switch_detected"] = switch_at is not None
        for m in members:
            member_fused = dict(fused)
            member_fused["frame_results"] = frame_results_by_tid[m["track_id"]]
            results[m["track_id"]] = member_fused
    return results


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Jersey Identity Phase")
    parser.add_argument("--sequence_dir", type=str, required=True)
    parser.add_argument("--detections_json", type=str, required=True)
    parser.add_argument("--team_assignments_json", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--inference_mode", type=str, default="temporal",
                        choices=["mil", "temporal", "hybrid"],
                        help="mil=top-K MIL; temporal=per-frame + fusion; hybrid=both")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--min_frames", type=int, default=8)
    parser.add_argument("--min_quality", type=float, default=0.0)
    parser.add_argument("--p1_threshold", type=float, default=0.85)
    parser.add_argument("--margin_threshold", type=float, default=0.20)
    parser.add_argument("--roster_json", type=str, default=None)
    parser.add_argument("--team_mapping", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--legibility_model", type=str, default=None)
    parser.add_argument("--legibility_threshold", type=float, default=0.5)
    parser.add_argument("--multi_crop", action="store_true", help="Enable multi-crop candidates")
    parser.add_argument("--min_legible_frames", type=int, default=4, help="Minimum legible frames required to filter tracklet")
    parser.add_argument("--min_peak_quality", type=float, default=0.3, help="Minimum quality score for a peak")
    parser.add_argument("--fusion_mode", type=str, default="geometric",
                        choices=["geometric", "arithmetic", "topk_geometric", "confidence_topk"],
                        help="Temporal fusion strategy (see temporal_fusion)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Per-frame probability temperature before fusion")
    parser.add_argument("--link_fragments", action="store_true",
                        help="Link ByteTrack fragments of the same player (team + temporal gap + "
                             "spatial continuity) and pool their evidence before fusion")
    parser.add_argument("--split_on_switch", action="store_true",
                        help="Detect probable ID switches inside tracklets; fuse only the dominant "
                             "segment and never lock across a switch")
    parser.add_argument("--reassign_conflicts", action="store_true",
                        help="On duplicate-number conflicts, losers fall back to their best "
                             "non-conflicting alternative (intra-frame exclusivity) instead of unknown")
    parser.add_argument("--parseq_model", type=str, default=None,
                        help="ensamble con PARSeq (scene-text): 'parseq' (base) "
                             "o 'parseq_tiny'; requiere descarga torch.hub")
    parser.add_argument("--parseq_weight", type=float, default=0.25)
    parser.add_argument("--parseq_checkpoint", type=str, default=None,
                        help="pesos fine-tuneados (runs/parseq_jersey_ft/best.pt) "
                             "a cargar sobre --parseq_model")
    parser.add_argument("--infer_unknowns", action="store_true",
                        help="ELIMINACION con roster: tracklets sin lock pero con evidencia "
                             "reciben el mejor numero del roster no usado por companeros "
                             "solapados (estado 'inferred', se muestra con '?')")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model = load_jersey_model(args.model_path, device, strict=True)

    legibility_model = None
    if args.legibility_model:
        print(f"Loading legibility model from {args.legibility_model}...")
        legibility_model = load_legibility_model(args.legibility_model, device)

    print("Building tracklet frames...")
    tracklets_data = build_tracklet_frames(
        args.detections_json,
        args.team_assignments_json,
        args.sequence_dir,
        K=args.K,
        min_frames=args.min_frames,
        min_quality=args.min_quality,
    )
    print(f"  {len(tracklets_data)} tracklets with >= {args.min_frames} frames")

    # --- Roster loading ---
    team_rosters = None
    if args.roster_json:
        seq_name = Path(args.sequence_dir).name
        with open(args.roster_json) as f:
            all_rosters = json.load(f)
        team_mapping = {}
        if args.team_mapping:
            mapping_path = Path(args.team_mapping)
            if mapping_path.exists() and mapping_path.suffix == ".json":
                with open(mapping_path) as f:
                    audit = json.load(f)
                team_mapping = {int(k): v for k, v in audit.get("team_mapping", {}).items()}
            else:
                for pair in args.team_mapping.split(","):
                    k, v = pair.strip().split(":")
                    team_mapping[int(k)] = v.strip()
        else:
            out_dir = Path(args.output_json).parent
            audit_path = out_dir / "team_audit.json"
            if audit_path.exists():
                with open(audit_path) as f:
                    audit = json.load(f)
                team_mapping = {int(k): v for k, v in audit.get("team_mapping", {}).items()}

        if team_mapping and seq_name in all_rosters:
            seq_roster = all_rosters[seq_name]
            team_rosters = {}
            for tid, side in team_mapping.items():
                side_data = seq_roster.get(side, {})
                numbers = set(side_data.get("field_players", []) + side_data.get("goalkeepers", []))
                if numbers:
                    team_rosters[tid] = numbers
            print(f"  Roster loaded: {{{', '.join(f'{k}: {len(v)} nums' for k, v in team_rosters.items())}}}")

    # --- Inference ---
    if args.inference_mode == "mil":
        print("Predicting with MIL (top-K)...")
        predictions = predict_tracklets_mil(model, tracklets_data, args.sequence_dir, device, K=args.K, seed=args.seed)
    elif args.inference_mode == "temporal":
        print("Predicting with temporal fusion (all frames)...")
        parseq = parseq_tf = None
        if args.parseq_model:
            print(f"Loading PARSeq ensemble reader: {args.parseq_model}")
            parseq, parseq_tf = load_parseq(args.parseq_model, device,
                                            checkpoint=args.parseq_checkpoint)
        predictions = predict_tracklets_temporal(
            model, tracklets_data, args.sequence_dir, device,
            p1_threshold=args.p1_threshold,
            margin_threshold=args.margin_threshold,
            team_rosters=team_rosters,
            legibility_model=legibility_model,
            legibility_threshold=args.legibility_threshold,
            multi_crop=args.multi_crop,
            min_legible_frames=args.min_legible_frames,
            min_peak_quality=args.min_peak_quality,
            fusion_mode=args.fusion_mode,
            temperature=args.temperature,
            do_link_fragments=args.link_fragments,
            split_on_switch=args.split_on_switch,
            parseq=parseq, parseq_tf=parseq_tf,
            parseq_weight=args.parseq_weight,
        )
    else:  # hybrid
        print("Predicting with MIL + temporal fusion...")
        mil_preds = predict_tracklets_mil(model, tracklets_data, args.sequence_dir, device, K=args.K, seed=args.seed)
        temporal_preds = predict_tracklets_temporal(
            model, tracklets_data, args.sequence_dir, device,
            p1_threshold=args.p1_threshold,
            margin_threshold=args.margin_threshold,
            team_rosters=team_rosters,
            legibility_model=legibility_model,
            legibility_threshold=args.legibility_threshold,
            multi_crop=args.multi_crop,
            min_legible_frames=args.min_legible_frames,
            min_peak_quality=args.min_peak_quality,
            fusion_mode=args.fusion_mode,
            temperature=args.temperature,
            do_link_fragments=args.link_fragments,
            split_on_switch=args.split_on_switch,
        )
        # For hybrid, default to temporal but include MIL as fallback reference
        predictions = temporal_preds
        for tid in predictions:
            predictions[tid]["mil_probs"] = mil_preds.get(tid, {}).get("probs")

    # --- Build TrackletInfo objects ---
    tracklet_infos = []
    for t in tracklets_data:
        tid = t["track_id"]
        pred = predictions.get(tid, {})
        probs = pred.get("probs")
        frame_ids = t.get("all_frame_ids", t.get("frame_ids", []))
        ti = TrackletInfo(
            track_id=tid,
            team_id=t["team_id"],
            jersey_probs=probs,
            num_frames=t["num_frames"],
            quality_scores=t.get("all_quality_scores", t.get("quality_scores", [])),
            state="unknown",
            frame_ids=frame_ids,
        )
        ti.alternatives = pred.get("alternatives", [])
        tracklet_infos.append(ti)

    print("Assigning jersey numbers per team...")
    assignments = assign_per_team(
        tracklet_infos,
        team_rosters=team_rosters,
        p1_threshold=args.p1_threshold,
        margin_threshold=args.margin_threshold,
        reassign_conflicts=args.reassign_conflicts,
        infer_by_elimination=args.infer_unknowns,
    )

    def _convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    tracklet_data_lookup = {t["track_id"]: t for t in tracklets_data}

    output = {
        "tracklets": [],
        "summary": {
            "inference_mode": args.inference_mode,
            "total": len(assignments),
            "locked": 0, "tentative": 0, "unknown": 0,
            "config": {
                "model_path": str(args.model_path),
                "legibility_model": str(args.legibility_model) if args.legibility_model else None,
                "legibility_threshold": float(args.legibility_threshold),
                "p1_threshold": float(args.p1_threshold),
                "margin_threshold": float(args.margin_threshold),
                "multi_crop": bool(args.multi_crop),
                "min_legible_frames": int(args.min_legible_frames),
                "min_peak_quality": float(args.min_peak_quality),
                "fusion_mode": str(args.fusion_mode),
                "temperature": float(args.temperature),
                "link_fragments": bool(args.link_fragments),
                "split_on_switch": bool(args.split_on_switch),
                "reassign_conflicts": bool(args.reassign_conflicts),
                "seed": int(args.seed),
            }
        }
    }

    for tid, asn in assignments.items():
        tdata = tracklet_data_lookup.get(tid, {})
        pred = predictions.get(tid, {})

        tracklet_out = {
            "track_id": _convert(tid),
            "sequence": Path(args.sequence_dir).name,
            "team_id": _convert(asn["team_id"]),
            "predicted_number": _convert(asn["predicted_number"]) if asn.get("predicted_number") is not None else None,
            "confidence": _convert(asn["confidence"]),
            "state": str(asn["state"]),
            "p1": _convert(asn.get("p1", 0.0)),
            "margin": _convert(asn.get("margin", 0.0)),
            "is_goalkeeper": bool(asn.get("is_goalkeeper", False)),
            "duplicate_conflict_with": _convert(asn.get("duplicate_conflict_with")) if asn.get("duplicate_conflict_with") is not None else None,
            "alternatives": getattr(next((ti for ti in tracklet_infos if ti.track_id == tid), None), "alternatives", []),
            "frame_ids": tdata.get("all_frame_ids", []),
            "start_frame": min(tdata.get("all_frame_ids", [0])) if tdata.get("all_frame_ids") else None,
            "end_frame": max(tdata.get("all_frame_ids", [0])) if tdata.get("all_frame_ids") else None,
            "inference_frame_ids": tdata.get("frame_ids", []),
            "num_frames_processed": pred.get("num_frames_processed", len(tdata.get("all_frame_ids", []))),
        }

        if args.link_fragments:
            tracklet_out["linked_group"] = _convert(pred.get("linked_group"))
            tracklet_out["linked_members"] = [_convert(m) for m in pred.get("linked_members", [])]
        if args.split_on_switch:
            tracklet_out["switch_detected"] = bool(pred.get("switch_detected", False))

        # Add temporal-specific fields
        if args.inference_mode in ("temporal", "hybrid"):
            tracklet_out["peaks"] = [
                {
                    "frame_id": p["frame_id"],
                    "quality": p["quality"],
                    "predicted_number": p["predicted_number"],
                    "confidence": p["confidence"],
                    "crop_variant": p.get("crop_variant", "torso_actual"),
                    "legibility_score": p.get("legibility_score", 1.0),
                }
                for p in pred.get("peaks", [])
            ]
            tracklet_out["peak_scores"] = {int(k): float(v) for k, v in pred.get("peak_scores", {}).items()}
            tracklet_out["legibility_score"] = _convert(np.mean([fr.get("legibility_score", 1.0) for fr in pred.get("frame_results", [])])) if pred.get("frame_results") else 1.0
            tracklet_out["frame_results"] = [
                {
                    "frame_id": fr["frame_id"],
                    "predicted_number": int(np.argmax(fr["jersey_probs"])) + 1,
                    "confidence": float(np.max(fr["jersey_probs"])),
                    "quality_score": fr["quality_score"],
                    "legibility_score": fr.get("legibility_score", 1.0),
                    "crop_variant": fr.get("crop_variant", "torso_actual"),
                }
                for fr in pred.get("frame_results", [])
            ]

        output["tracklets"].append(tracklet_out)

        if asn["state"] == "locked":
            output["summary"]["locked"] += 1
        elif asn["state"] == "tentative":
            output["summary"]["tentative"] += 1
        else:
            output["summary"]["unknown"] += 1

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {args.output_json}")
    print(f"  locked: {output['summary']['locked']}, tentative: {output['summary']['tentative']}, unknown: {output['summary']['unknown']}")


if __name__ == "__main__":
    main()
